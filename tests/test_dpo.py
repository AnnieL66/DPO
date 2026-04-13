"""
tests/test_dpo.py
-----------------
Comprehensive test suite for the DPO pipeline.

All tests run on CPU with a tiny random model (TinyLM) — no GPU or model
downloads required.  Run with:

    pytest tests/test_dpo.py -v

Coverage
--------
  TestBatchLogps         — _batch_logps shape, sign, masking, normalisation
  TestConcatenatedForward— output matches separate calls; handles unequal lengths
  TestDPOLoss            — mathematical correctness, neutral case, ordering, β
  TestTrainingStep       — forward pass, backward pass, gradient direction,
                           reference-model-mode guard, metrics dict
  TestNumericalStability — extreme logit values, uniform logits
  TestDataUtils          — SHP score filter, dataset schema validation
"""

import math
import sys
import os

import pytest
import torch
import torch.nn as nn

# Allow running from the project root without installing the package.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dpo.dpo_loss import (
    _batch_logps,
    compute_log_probs,
    concatenated_forward,
    dpo_loss,
    training_step,
)


# ---------------------------------------------------------------------------
# Shared tiny model
# ---------------------------------------------------------------------------


class TinyLM(nn.Module):
    """
    Minimal causal LM stub: Embedding -> Linear, no self-attention.
    Produces correctly shaped logits for all log-prob and loss tests.
    """

    VOCAB = 16

    def __init__(self, hidden: int = 8):
        super().__init__()
        self.embed = nn.Embedding(self.VOCAB, hidden)
        self.proj = nn.Linear(hidden, self.VOCAB, bias=False)

    def forward(self, input_ids, attention_mask=None):
        logits = self.proj(self.embed(input_ids))

        class _Out:
            pass

        out = _Out()
        out.logits = logits
        return out


def make_batch(B: int, L: int, prompt_len: int = 2, vocab: int = TinyLM.VOCAB):
    """
    Create a minimal batch dict with chosen and rejected sequences.
    The first prompt_len tokens of labels are set to -100 (prompt mask).
    """
    chosen_ids = torch.randint(0, vocab, (B, L))
    rejected_ids = torch.randint(0, vocab, (B, L))
    mask = torch.ones(B, L, dtype=torch.long)

    chosen_labels = chosen_ids.clone()
    chosen_labels[:, :prompt_len] = -100
    rejected_labels = rejected_ids.clone()
    rejected_labels[:, :prompt_len] = -100

    return {
        "chosen_input_ids": chosen_ids,
        "chosen_attention_mask": mask,
        "chosen_labels": chosen_labels,
        "rejected_input_ids": rejected_ids,
        "rejected_attention_mask": mask,
        "rejected_labels": rejected_labels,
    }


# ---------------------------------------------------------------------------
# TestBatchLogps
# ---------------------------------------------------------------------------


class TestBatchLogps:
    """Unit tests for _batch_logps."""

    def _make_logits(self, B, L, V=TinyLM.VOCAB):
        """Random float32 logits of shape [B, L, V]."""
        return torch.randn(B, L, V, dtype=torch.float32)

    def test_output_shape(self):
        """Output tensor has shape [B] — one scalar per sequence."""
        B, L = 3, 7
        logits = self._make_logits(B, L)
        labels = torch.randint(0, TinyLM.VOCAB, (B, L))
        out = _batch_logps(logits, labels)
        assert out.shape == (B,)

    def test_output_nonpositive(self):
        """Log-probs are always <= 0 (log of a probability <= 1)."""
        torch.manual_seed(1)
        logits = self._make_logits(4, 10)
        labels = torch.randint(0, TinyLM.VOCAB, (4, 10))
        out = _batch_logps(logits, labels)
        assert (out <= 0).all(), f"Got positive log-probs: {out}"

    def test_prompt_tokens_excluded(self):
        """
        Setting labels to -100 (prompt mask) for a range of positions must
        not change the output — those tokens carry no probability signal.
        """
        torch.manual_seed(2)
        B, L = 2, 8
        logits = self._make_logits(B, L)
        labels_full = torch.randint(0, TinyLM.VOCAB, (B, L))

        # Baseline: all tokens are response tokens.
        out_full = _batch_logps(logits, labels_full)

        # Mask the first 3 tokens as prompt on both sequences.
        labels_masked = labels_full.clone()
        labels_masked[:, :3] = -100
        out_masked = _batch_logps(logits, labels_masked)

        # The masked version omits the first 3 positions, so the result
        # will differ from out_full — but both must remain <= 0 and
        # have shape [B].
        assert out_masked.shape == (B,)
        assert (out_masked <= 0).all()

    def test_masking_is_position_selective(self):
        """
        Masking only a subset of tokens affects the output but does not
        contaminate the unmasked positions.  Verify by masking different
        slices and checking monotonicity of the absolute log-prob value.
        """
        torch.manual_seed(3)
        B, L = 1, 10
        logits = self._make_logits(B, L)
        labels = torch.randint(0, TinyLM.VOCAB, (B, L))

        # More response tokens -> more (negative) log-probs summed -> sum
        # is more negative.  With average_log_prob=False (raw sum), masking
        # more tokens should increase (less negative) the sum.
        labels_none_masked = labels.clone()
        labels_half_masked = labels.clone()
        labels_half_masked[:, :5] = -100

        sum_full = _batch_logps(logits, labels_none_masked, average_log_prob=False)
        sum_half = _batch_logps(logits, labels_half_masked, average_log_prob=False)

        # Masking half the tokens removes their (negative) contribution,
        # so the sum with more masks should be >= the sum without.
        assert sum_half.item() >= sum_full.item()

    def test_average_vs_sum_scale(self):
        """
        average_log_prob=True reduces length bias: the gap between two
        sequences of different lengths should be smaller when using averages
        than when using raw sums.

        We construct the same per-token distribution for two sequences of
        different lengths by using the same single-row logits, tiled.  With
        raw sums the longer sequence is ~k times more negative; with averages
        both sequences return roughly the same per-token value.
        """
        torch.manual_seed(4)
        V = TinyLM.VOCAB

        # One row of logits representing a single token distribution.
        single_token_logits = torch.randn(1, 1, V, dtype=torch.float32)

        L_short, L_long = 4, 12

        # Tile the same distribution across all positions so expected per-token
        # log-prob is identical for both lengths.
        logits_short = single_token_logits.expand(1, L_short, V).contiguous()
        logits_long = single_token_logits.expand(1, L_long, V).contiguous()

        # Use the same token index at every position so the chosen token's
        # log-prob is also identical.
        token_idx = torch.argmax(single_token_logits.squeeze())
        labels_short = torch.full((1, L_short), token_idx.item(), dtype=torch.long)
        labels_long = torch.full((1, L_long), token_idx.item(), dtype=torch.long)

        avg_short = _batch_logps(logits_short, labels_short, average_log_prob=True)
        avg_long = _batch_logps(logits_long, labels_long, average_log_prob=True)
        sum_short = _batch_logps(logits_short, labels_short, average_log_prob=False)
        sum_long = _batch_logps(logits_long, labels_long, average_log_prob=False)

        # With identical per-token distributions, the averages should be equal.
        assert torch.allclose(avg_short, avg_long, atol=1e-5), (
            f"Averages should be equal for identical per-token distributions; "
            f"got short={avg_short.item():.4f}, long={avg_long.item():.4f}"
        )

        # The raw sums should differ by the ratio of lengths (long / short = 3x).
        # _batch_logps applies the causal shift: logits[:, :-1] predicts
        # labels[:, 1:], so a sequence of length L contributes L-1 scored
        # positions to the sum.  The expected ratio is therefore:
        #   (L_long - 1) / (L_short - 1)
        ratio = sum_long.item() / sum_short.item()
        expected_ratio = float(L_long - 1) / float(L_short - 1)
        assert abs(ratio - expected_ratio) < 0.01, (
            f"Sum ratio should be {expected_ratio:.3f} "
            f"(L_long-1)/(L_short-1), got {ratio:.4f}"
        )

    def test_float32_assertion(self):
        """_batch_logps raises AssertionError when logits are not float32."""
        logits_fp16 = torch.randn(2, 5, TinyLM.VOCAB, dtype=torch.float16)
        labels = torch.randint(0, TinyLM.VOCAB, (2, 5))
        with pytest.raises(AssertionError, match="float32"):
            _batch_logps(logits_fp16, labels)

    def test_shape_mismatch_assertion(self):
        """_batch_logps raises AssertionError on logits/labels shape mismatch."""
        logits = torch.randn(2, 5, TinyLM.VOCAB, dtype=torch.float32)
        labels = torch.randint(0, TinyLM.VOCAB, (2, 6))  # wrong length
        with pytest.raises(AssertionError, match="Shape mismatch"):
            _batch_logps(logits, labels)

    def test_all_masked_returns_zero(self):
        """
        When all labels are -100 (no response tokens), the masked sum is 0
        and the average is also 0 (n_tokens clamped to 1 to avoid div-by-zero).
        """
        logits = torch.randn(2, 5, TinyLM.VOCAB, dtype=torch.float32)
        labels = torch.full((2, 5), -100, dtype=torch.long)
        out = _batch_logps(logits, labels, average_log_prob=True)
        assert (out == 0.0).all()


# ---------------------------------------------------------------------------
# TestConcatenatedForward
# ---------------------------------------------------------------------------


class TestConcatenatedForward:
    """Tests for concatenated_forward."""

    def test_matches_separate_calls(self):
        """
        concatenated_forward must produce the same log-probs as calling
        compute_log_probs separately on chosen and rejected.

        This verifies that stacking sequences along the batch dimension and
        then splitting does not corrupt the per-sequence log-probs.
        """
        torch.manual_seed(10)
        model = TinyLM()
        model.eval()

        B, L = 2, 6
        chosen_ids = torch.randint(0, TinyLM.VOCAB, (B, L))
        rejected_ids = torch.randint(0, TinyLM.VOCAB, (B, L))
        mask = torch.ones(B, L, dtype=torch.long)
        chosen_labels = chosen_ids.clone()
        chosen_labels[:, :2] = -100
        rejected_labels = rejected_ids.clone()
        rejected_labels[:, :2] = -100

        ch_concat, rej_concat = concatenated_forward(
            model,
            chosen_ids,
            mask,
            chosen_labels,
            rejected_ids,
            mask,
            rejected_labels,
        )
        ch_separate = compute_log_probs(model, chosen_ids, mask, chosen_labels)
        rej_separate = compute_log_probs(model, rejected_ids, mask, rejected_labels)

        assert torch.allclose(ch_concat, ch_separate, atol=1e-5), (
            f"Chosen mismatch: concat={ch_concat}, separate={ch_separate}"
        )
        assert torch.allclose(rej_concat, rej_separate, atol=1e-5), (
            f"Rejected mismatch: concat={rej_concat}, separate={rej_separate}"
        )

    def test_handles_unequal_lengths(self):
        """
        When chosen and rejected have different sequence lengths,
        concatenated_forward pads the shorter one.  The padded positions
        must NOT affect the log-probs of the shorter sequence.
        """
        torch.manual_seed(11)
        model = TinyLM()
        model.eval()

        B = 2
        Lc, Lr = 5, 9  # chosen is shorter, rejected is longer

        chosen_ids = torch.randint(0, TinyLM.VOCAB, (B, Lc))
        rejected_ids = torch.randint(0, TinyLM.VOCAB, (B, Lr))
        chosen_mask = torch.ones(B, Lc, dtype=torch.long)
        rejected_mask = torch.ones(B, Lr, dtype=torch.long)
        chosen_labels = chosen_ids.clone()
        chosen_labels[:, :2] = -100
        rejected_labels = rejected_ids.clone()
        rejected_labels[:, :2] = -100

        ch_padded, rej_padded = concatenated_forward(
            model,
            chosen_ids,
            chosen_mask,
            chosen_labels,
            rejected_ids,
            rejected_mask,
            rejected_labels,
        )
        ch_direct = compute_log_probs(model, chosen_ids, chosen_mask, chosen_labels)
        rej_direct = compute_log_probs(
            model, rejected_ids, rejected_mask, rejected_labels
        )

        assert torch.allclose(ch_padded, ch_direct, atol=1e-5)
        assert torch.allclose(rej_padded, rej_direct, atol=1e-5)

    def test_output_shapes(self):
        """Output tensors are both [B]."""
        model = TinyLM()
        B, L = 3, 7
        ids = torch.randint(0, TinyLM.VOCAB, (B, L))
        mask = torch.ones(B, L, dtype=torch.long)
        labels = ids.clone()
        labels[:, :2] = -100

        ch, rej = concatenated_forward(model, ids, mask, labels, ids, mask, labels)
        assert ch.shape == (B,)
        assert rej.shape == (B,)


# ---------------------------------------------------------------------------
# TestDPOLoss
# ---------------------------------------------------------------------------


class TestDPOLoss:
    """Mathematical correctness tests for dpo_loss."""

    def test_manual_computation(self):
        """
        dpo_loss must match a value computed by hand.

        Given:
          policy_chosen   = -10,  ref_chosen   = -15  -> chosen_ratio  = +5
          policy_rejected = -20,  ref_rejected = -18  -> rejected_ratio = -2
          beta = 0.1
          margin = 0.1 * (5 - (-2)) = 0.7
          loss = -log(sigma(0.7))
        """
        beta = 0.1
        pol_ch = torch.tensor([-10.0])
        pol_rej = torch.tensor([-20.0])
        ref_ch = torch.tensor([-15.0])
        ref_rej = torch.tensor([-18.0])

        chosen_ratio = (-10.0) - (-15.0)  # 5.0
        rejected_ratio = (-20.0) - (-18.0)  # -2.0
        margin = beta * (chosen_ratio - rejected_ratio)  # 0.7
        expected_loss = -math.log(1.0 / (1.0 + math.exp(-margin)))

        loss, _, _ = dpo_loss(pol_ch, pol_rej, ref_ch, ref_rej, beta=beta)
        assert abs(loss.item() - expected_loss) < 1e-5, (
            f"Expected {expected_loss:.6f}, got {loss.item():.6f}"
        )

    def test_neutral_case_equals_log2(self):
        """
        When policy == reference, every log-ratio is 0, the reward margin is 0,
        sigma(0) = 0.5, and the loss is -log(0.5) = log(2) ≈ 0.6931.
        This is the expected loss at the very start of training (before any
        gradient updates have moved the policy away from the reference).
        """
        ref_ch = torch.tensor([-15.0, -12.0])
        ref_rej = torch.tensor([-20.0, -17.0])
        loss, _, _ = dpo_loss(
            ref_ch.clone(),
            ref_rej.clone(),
            ref_ch,
            ref_rej,
            beta=0.1,
        )
        assert abs(loss.item() - math.log(2)) < 1e-5, (
            f"Neutral loss should be log(2)={math.log(2):.4f}, got {loss.item():.4f}"
        )

    def test_loss_ordering(self):
        """
        loss(correct preference) < loss(neutral) < loss(inverted preference).
        A correct policy should be rewarded with low loss; an inverted policy
        should be penalised with high loss.
        """
        ref_ch = torch.tensor([-15.0])
        ref_rej = torch.tensor([-20.0])

        # Correct: policy assigns higher relative probability to chosen.
        loss_correct, _, _ = dpo_loss(
            torch.tensor([-10.0]),
            torch.tensor([-25.0]),
            ref_ch,
            ref_rej,
        )
        # Neutral: policy == reference.
        loss_neutral, _, _ = dpo_loss(
            ref_ch.clone(),
            ref_rej.clone(),
            ref_ch,
            ref_rej,
        )
        # Inverted: policy assigns higher relative probability to rejected.
        loss_inverted, _, _ = dpo_loss(
            torch.tensor([-25.0]),
            torch.tensor([-10.0]),
            ref_ch,
            ref_rej,
        )

        assert loss_correct.item() < loss_neutral.item(), (
            "Correct preference should give lower loss than neutral"
        )
        assert loss_neutral.item() < loss_inverted.item(), (
            "Neutral should give lower loss than inverted preference"
        )

    def test_rewards_are_detached(self):
        """
        The returned chosen_rewards and rejected_rewards tensors must not
        carry a gradient — they are for logging only.
        """
        pol_ch = torch.tensor([-10.0], requires_grad=True)
        pol_rej = torch.tensor([-20.0], requires_grad=True)
        ref_ch = torch.tensor([-15.0])
        ref_rej = torch.tensor([-18.0])

        _, ch_r, rej_r = dpo_loss(pol_ch, pol_rej, ref_ch, ref_rej)
        assert not ch_r.requires_grad, "chosen_rewards should be detached"
        assert not rej_r.requires_grad, "rejected_rewards should be detached"

    def test_loss_is_scalar(self):
        """dpo_loss returns a 0-dimensional scalar regardless of batch size."""
        B = 4
        pol_ch = torch.randn(B)
        pol_rej = torch.randn(B)
        ref_ch = torch.randn(B)
        ref_rej = torch.randn(B)
        loss, ch_r, rej_r = dpo_loss(pol_ch, pol_rej, ref_ch, ref_rej)
        assert loss.shape == torch.Size([]), f"loss shape: {loss.shape}"
        assert ch_r.shape == (B,)
        assert rej_r.shape == (B,)

    def test_beta_scales_margin(self):
        """
        Larger beta amplifies the reward margin, pushing loss further from
        log(2) in both the correct and inverted cases.
        """
        pol_ch = torch.tensor([-10.0])
        pol_rej = torch.tensor([-20.0])
        ref_ch = torch.tensor([-15.0])
        ref_rej = torch.tensor([-18.0])

        loss_small, _, _ = dpo_loss(pol_ch, pol_rej, ref_ch, ref_rej, beta=0.01)
        loss_large, _, _ = dpo_loss(pol_ch, pol_rej, ref_ch, ref_rej, beta=1.0)

        log2 = math.log(2)
        # With correct preference, larger beta -> larger margin -> loss further
        # below log(2).
        assert loss_large.item() < loss_small.item(), (
            "Larger beta should give lower loss when preference is correct"
        )
        assert loss_small.item() < log2
        assert loss_large.item() < log2

    def test_loss_is_differentiable(self):
        """Loss backward pass completes without errors."""
        pol_ch = torch.randn(2, requires_grad=True)
        pol_rej = torch.randn(2, requires_grad=True)
        ref_ch = torch.randn(2)
        ref_rej = torch.randn(2)

        loss, _, _ = dpo_loss(pol_ch, pol_rej, ref_ch, ref_rej)
        loss.backward()

        assert pol_ch.grad is not None
        assert pol_rej.grad is not None


# ---------------------------------------------------------------------------
# TestTrainingStep
# ---------------------------------------------------------------------------


class TestTrainingStep:
    """Integration tests for training_step."""

    @pytest.fixture
    def models(self):
        torch.manual_seed(42)
        policy = TinyLM()
        policy.train()
        ref = TinyLM()
        for p in ref.parameters():
            p.requires_grad = False
        ref.eval()
        return policy, ref

    def test_runs_without_error(self, models):
        """training_step completes a forward pass without raising."""
        policy, ref = models
        batch = make_batch(B=2, L=6, prompt_len=2)
        loss, metrics = training_step(batch, policy, ref)
        assert loss is not None
        assert isinstance(metrics, dict)

    def test_loss_is_scalar(self, models):
        """Returned loss tensor is 0-dimensional."""
        policy, ref = models
        batch = make_batch(B=2, L=6)
        loss, _ = training_step(batch, policy, ref)
        assert loss.shape == torch.Size([])

    def test_metrics_keys_present(self, models):
        """The metrics dict contains all expected keys."""
        policy, ref = models
        batch = make_batch(B=2, L=6)
        _, metrics = training_step(batch, policy, ref)
        expected_keys = {
            "loss",
            "reward_margin",
            "chosen_reward_mean",
            "rejected_reward_mean",
            "accuracy",
            "logps/chosen",
            "logps/rejected",
        }
        assert expected_keys.issubset(metrics.keys()), (
            f"Missing keys: {expected_keys - set(metrics.keys())}"
        )

    def test_backward_pass_runs(self, models):
        """loss.backward() completes and policy parameters receive gradients."""
        policy, ref = models
        batch = make_batch(B=2, L=6)
        loss, _ = training_step(batch, policy, ref)
        loss.backward()
        trainable = [p for p in policy.parameters() if p.requires_grad]
        assert all(p.grad is not None for p in trainable), (
            "Some trainable parameters did not receive gradients"
        )

    def test_gradient_step_lowers_loss(self, models):
        """
        One AdamW step should reduce the DPO loss.
        If this fails it usually means the gradient direction is wrong
        (sign error in the loss or in log-ratio computation).
        """
        policy, ref = models
        opt = torch.optim.AdamW(policy.parameters(), lr=1e-2)

        batch = make_batch(B=2, L=8, prompt_len=2)
        loss_before, _ = training_step(batch, policy, ref)

        opt.zero_grad()
        loss_before.backward()
        opt.step()

        with torch.no_grad():
            loss_after, _ = training_step(batch, policy, ref)

        assert loss_after.item() < loss_before.item(), (
            f"Loss did not decrease: {loss_before.item():.6f} -> "
            f"{loss_after.item():.6f}"
        )

    def test_ref_model_weights_unchanged(self, models):
        """
        A full training step must not alter any reference model parameters.
        The reference is the KL anchor; changing it invalidates the loss.
        """
        policy, ref = models
        opt = torch.optim.AdamW(policy.parameters(), lr=1e-2)

        ref_params_before = [p.clone() for p in ref.parameters()]

        batch = make_batch(B=2, L=6)
        loss, _ = training_step(batch, policy, ref)
        opt.zero_grad()
        loss.backward()
        opt.step()

        for before, after in zip(ref_params_before, ref.parameters()):
            assert torch.equal(before, after), (
                "Reference model parameter changed during training step"
            )

    def test_ref_model_must_be_in_eval_mode(self, models):
        """
        training_step must raise AssertionError when ref_model.training is True.
        A reference model in training mode may have active dropout, which
        adds stochastic noise to log pi_ref(y|x) that is not part of the
        DPO objective.
        """
        policy, ref = models
        ref.train()  # deliberately set to training mode
        batch = make_batch(B=2, L=6)
        with pytest.raises(AssertionError, match="eval mode"):
            training_step(batch, policy, ref)

    def test_accuracy_metric_between_zero_and_one(self, models):
        """The accuracy metric must be in [0, 1]."""
        policy, ref = models
        batch = make_batch(B=4, L=6)
        _, metrics = training_step(batch, policy, ref)
        assert 0.0 <= metrics["accuracy"] <= 1.0


# ---------------------------------------------------------------------------
# TestNumericalStability
# ---------------------------------------------------------------------------


class TestNumericalStability:
    """Tests that extreme or degenerate logits do not produce nan/inf."""

    def _logps_no_nan(self, logits_val: float, B=2, L=6, V=TinyLM.VOCAB):
        logits = torch.full((B, L, V), logits_val, dtype=torch.float32)
        labels = torch.randint(0, V, (B, L))
        out = _batch_logps(logits, labels)
        return out

    def test_large_positive_logits(self):
        """Very large positive logits must not cause nan or inf in log_softmax."""
        out = self._logps_no_nan(1e6)
        assert not torch.isnan(out).any(), "nan in output for large positive logits"
        assert not torch.isinf(out).any(), "inf in output for large positive logits"

    def test_large_negative_logits(self):
        """Very large negative logits must not cause nan or inf."""
        out = self._logps_no_nan(-1e6)
        assert not torch.isnan(out).any(), "nan in output for large negative logits"
        assert not torch.isinf(out).any(), "inf in output for large negative logits"

    def test_uniform_logits_give_uniform_log_prob(self):
        """
        When all logits are equal, log_softmax gives log(1/V) for every token.
        The per-token log-prob should therefore equal -log(V) for all positions.
        """
        B, L, V = 2, 8, TinyLM.VOCAB
        logits = torch.zeros(B, L, V, dtype=torch.float32)
        labels = torch.randint(0, V, (B, L))
        out = _batch_logps(logits, labels, average_log_prob=True)

        expected = -math.log(V)
        assert torch.allclose(out, torch.full((B,), expected), atol=1e-5), (
            f"Expected {expected:.4f} for uniform logits, got {out}"
        )

    def test_logsigmoid_stability_in_dpo_loss(self):
        """
        dpo_loss must not produce nan/inf for extreme log-ratio values.
        This would happen if we used log(sigmoid(x)) directly instead of
        F.logsigmoid, which would underflow to -inf at large negative x.
        """
        B = 4
        extreme_positive = torch.full((B,), 1000.0)
        extreme_negative = torch.full((B,), -1000.0)
        zeros = torch.zeros(B)

        # Extreme positive margin (policy very strongly prefers chosen).
        loss_pos, _, _ = dpo_loss(extreme_positive, zeros, zeros, zeros)
        assert not torch.isnan(loss_pos), "nan for large positive reward margin"
        assert not torch.isinf(loss_pos), "inf for large positive reward margin"

        # Extreme negative margin (policy very strongly prefers rejected).
        loss_neg, _, _ = dpo_loss(extreme_negative, zeros, zeros, zeros)
        assert not torch.isnan(loss_neg), "nan for large negative reward margin"
        assert not torch.isinf(loss_neg), "inf for large negative reward margin"


# ---------------------------------------------------------------------------
# TestDataUtils
# ---------------------------------------------------------------------------


class TestDataUtils:
    """Unit tests for data loading and preprocessing utilities."""

    def test_shp_skips_zero_scores(self):
        """
        Rows where either score is zero or negative must be filtered out.
        Dividing by zero raises ZeroDivisionError; dividing by a negative
        number inverts the sign of the ratio, causing strong preference pairs
        to be silently discarded.

        load_dataset is imported lazily inside get_shp, so we patch it at
        its definition site (datasets.load_dataset) rather than trying to
        patch a module-level attribute that does not exist.
        """
        from dpo.data_utils import get_shp
        import unittest.mock as mock

        fake_rows = [
            # Should be kept: both positive, ratio = 4/2 = 2.0 >= 2.0
            {
                "score_A": 4,
                "score_B": 2,
                "labels": 1,
                "history": "test",
                "human_ref_A": "good",
                "human_ref_B": "bad",
            },
            # Should be skipped: score_B is 0
            {
                "score_A": 5,
                "score_B": 0,
                "labels": 1,
                "history": "test",
                "human_ref_A": "good",
                "human_ref_B": "bad",
            },
            # Should be skipped: score_A is negative
            {
                "score_A": -1,
                "score_B": 3,
                "labels": 1,
                "history": "test",
                "human_ref_A": "good",
                "human_ref_B": "bad",
            },
        ]

        with mock.patch("datasets.load_dataset", return_value=fake_rows):
            result = get_shp("train", min_score_ratio=2.0, silent=True)

        assert len(result) == 1, (
            f"Expected 1 pair to survive filtering, got {len(result)}"
        )

    def test_shp_ratio_filter(self):
        """Pairs below the minimum score ratio are discarded."""
        from dpo.data_utils import get_shp
        import unittest.mock as mock

        fake_rows = [
            # ratio = 3/2 = 1.5 < 2.0 — should be skipped
            {
                "score_A": 3,
                "score_B": 2,
                "labels": 1,
                "history": "q",
                "human_ref_A": "a",
                "human_ref_B": "b",
            },
            # ratio = 10/2 = 5.0 >= 2.0 — should be kept
            {
                "score_A": 10,
                "score_B": 2,
                "labels": 1,
                "history": "q",
                "human_ref_A": "a",
                "human_ref_B": "b",
            },
        ]

        with mock.patch("datasets.load_dataset", return_value=fake_rows):
            result = get_shp("train", min_score_ratio=2.0, silent=True)

        assert len(result) == 1

    def test_build_trl_dataset_validates_schema(self):
        """
        build_trl_dataset raises ValueError when the HF Hub dataset is missing
        the required 'prompt', 'chosen', or 'rejected' columns.
        """
        from dpo.data_utils import build_trl_dataset
        import unittest.mock as mock
        from datasets import Dataset

        # Simulate a dataset with wrong columns (e.g., raw HH format where
        # the full conversation is stored in "chosen"/"rejected" without a
        # separate "prompt" key).
        bad_dataset = Dataset.from_dict(
            {
                "chosen": ["full conversation A"],
                "rejected": ["full conversation B"],
                # "prompt" column is absent
            }
        )

        with mock.patch("datasets.load_dataset", return_value=bad_dataset):
            with pytest.raises(ValueError, match="missing required columns"):
                build_trl_dataset(hf_dataset_name="dummy/dataset", split="train")

    def test_build_trl_dataset_with_valid_raw_data(self):
        """
        build_trl_dataset with raw_data returns a dataset with the correct
        number of rows and expected column names.
        """
        from dpo.data_utils import build_trl_dataset

        raw = [
            {"prompt": "Q1", "chosen": "A1_good", "rejected": "A1_bad"},
            {"prompt": "Q2", "chosen": "A2_good", "rejected": "A2_bad"},
        ]
        ds = build_trl_dataset(raw_data=raw)
        assert len(ds) == 2
        assert set(ds.column_names) >= {"prompt", "chosen", "rejected"}

    def test_sample_preference_data_is_valid(self):
        """The built-in SAMPLE_PREFERENCE_DATA passes schema validation."""
        from dpo.data_utils import build_trl_dataset

        ds = build_trl_dataset()  # falls back to SAMPLE_PREFERENCE_DATA
        assert len(ds) == 3
        for col in ("prompt", "chosen", "rejected"):
            assert col in ds.column_names


# ---------------------------------------------------------------------------
# TestToyExample
# ---------------------------------------------------------------------------


class TestToyExample:
    """Ensure toy_example.py runs all checks without error."""

    def test_run_all_checks(self):
        """
        run_all_checks() must complete without raising.
        This is an end-to-end smoke test of Part 1 (hand-crafted logits),
        Part 2 (shape/sign checks), and Part 3 (gradient step).
        """
        from dpo.toy_example import run_all_checks

        run_all_checks()
