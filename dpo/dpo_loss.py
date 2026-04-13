"""
dpo_loss.py
--------------------
Core DPO mathematics for code generation.

Public API
    _batch_logps         — sequence-level log P(response | prompt) with optional
                           length normalisation (NEW: average_log_prob param)
    compute_log_probs    — thin wrapper for single-sequence callers
    concatenated_forward — ONE forward pass for chosen+rejected
    dpo_loss             — exact DPO objective from Rafailov et al. 2023
    training_step        — full forward+backward-ready step for one batch
"""

from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# 1. Pure log-prob computation
# ---------------------------------------------------------------------------


def _batch_logps(
    logits: Tensor,  # [B, L, V]  — MUST be float32
    labels: Tensor,  # [B, L]     — -100 for prompt / pad
    average_log_prob: bool = True,
) -> Tensor:  # [B]
    """
    Sequence-level log P(response | prompt) for a batch.

    average_log_prob=True (default, recommended for code generation):
        Returns mean per-token log-prob: sum(log_probs) / n_response_tokens.
        Sequences of different lengths are put on the same scale, so the
        DPO reward margin is driven by quality, not token count.

    average_log_prob=False (strict DPO-paper formulation):
        Returns the raw sum, identical to Eq. 7 of Rafailov et al. 2023.
        Correct in theory but biased when chosen/rejected differ in length.
    """
    assert logits.dtype == torch.float32, (
        f"logits must be float32 before _batch_logps; got {logits.dtype}. "
        "Cast with .to(torch.float32) after the model forward pass."
    )
    assert logits.shape[:-1] == labels.shape, (
        f"Shape mismatch: logits {logits.shape}, labels {labels.shape}"
    )

    # Causal-LM shift: logit at position t predicts token at position t+1.
    shift_logits = logits[:, :-1, :].contiguous()  # [B, L-1, V]
    shift_labels = labels[:, 1:].contiguous()  # [B, L-1]

    log_probs = F.log_softmax(shift_logits, dim=-1)  # [B, L-1, V]  (stable)

    # clamp(min=0): makes the gather valid even at -100 positions; those
    # positions are multiplied by 0 via response_mask below.
    token_log_probs = log_probs.gather(
        dim=-1,
        index=shift_labels.clamp(min=0).unsqueeze(-1),
    ).squeeze(-1)  # [B, L-1]

    # Mask: 1 for real response tokens, 0 for prompt or padding (-100).
    response_mask = (shift_labels != -100).float()  # [B, L-1]

    masked_sum = (token_log_probs * response_mask).sum(dim=-1)  # [B]

    if average_log_prob:
        n_tokens = response_mask.sum(dim=-1).clamp(min=1)  # [B]
        return masked_sum / n_tokens
    else:
        return masked_sum


# ---------------------------------------------------------------------------
# 2. Concatenated forward pass (key efficiency optimisation)
# ---------------------------------------------------------------------------


def concatenated_forward(
    model: torch.nn.Module,
    chosen_ids: Tensor,  # [B, Lc]
    chosen_mask: Tensor,  # [B, Lc]
    chosen_labels: Tensor,  # [B, Lc]
    rejected_ids: Tensor,  # [B, Lr]
    rejected_mask: Tensor,  # [B, Lr]
    rejected_labels: Tensor,  # [B, Lr]
    pad_token_id: int = 0,
    average_log_prob: bool = True,
) -> Tuple[Tensor, Tensor]:  # ([B], [B])
    """
    Run ONE forward pass with chosen and rejected stacked along batch dim.

    Chosen and rejected may have different lengths; the shorter one is
    right-padded to max_len using pad_token_id / attention_mask=0 /
    labels=-100 so padded positions contribute nothing to the log-prob.

    Why right-padding?
        For causal LMs, position t can only attend to positions 0..t.
        Right-padding means padding only appears *after* all real tokens,
        so it never enters the attention of any real token.

    pad_token_id should come from tokenizer.pad_token_id.  The actual
    value does not affect the loss (those positions are masked out), but
    using the correct ID avoids any model that inspects raw input IDs.
    """
    B = chosen_ids.shape[0]
    max_len = max(chosen_ids.shape[1], rejected_ids.shape[1])

    def _pad(t: Tensor, length: int, pad_val: int) -> Tensor:
        if t.shape[1] == length:
            return t
        pad = t.new_full((t.shape[0], length - t.shape[1]), pad_val)
        return torch.cat([t, pad], dim=1)

    all_ids = torch.cat(
        [
            _pad(chosen_ids, max_len, pad_token_id),
            _pad(rejected_ids, max_len, pad_token_id),
        ],
        dim=0,
    )
    all_mask = torch.cat(
        [_pad(chosen_mask, max_len, 0), _pad(rejected_mask, max_len, 0)],
        dim=0,
    )
    all_labels = torch.cat(
        [_pad(chosen_labels, max_len, -100), _pad(rejected_labels, max_len, -100)],
        dim=0,
    )

    # Cast to float32 BEFORE log_softmax.
    # bfloat16 has only 7 mantissa bits; log_softmax over Qwen's 152k vocab
    # requires the extra precision of float32 to avoid noisy log-ratios.
    all_logits = model(input_ids=all_ids, attention_mask=all_mask).logits.to(
        torch.float32
    )

    all_logps = _batch_logps(
        all_logits, all_labels, average_log_prob=average_log_prob
    )  # [2B]

    return all_logps[:B], all_logps[B:]


# ---------------------------------------------------------------------------
# 3. Convenience wrapper for single-sequence callers
# ---------------------------------------------------------------------------


def compute_log_probs(
    model: torch.nn.Module,
    input_ids: Tensor,  # [B, L]
    attention_mask: Tensor,  # [B, L]
    labels: Tensor,  # [B, L]  — -100 for prompt tokens
    average_log_prob: bool = True,
) -> Tensor:  # [B]
    """
    Thin wrapper for single-sequence use (toy_example, tests, evaluation).
    For training, prefer concatenated_forward — it halves forward calls.
    """
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.to(
        torch.float32
    )
    return _batch_logps(logits, labels, average_log_prob=average_log_prob)


# ---------------------------------------------------------------------------
# 4. DPO loss
# ---------------------------------------------------------------------------


def dpo_loss(
    policy_chosen_logps: Tensor,  # [B]  log πθ(y_w | x)
    policy_rejected_logps: Tensor,  # [B]  log πθ(y_l | x)
    ref_chosen_logps: Tensor,  # [B]  log πref(y_w | x)
    ref_rejected_logps: Tensor,  # [B]  log πref(y_l | x)
    beta: float = 0.1,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Exact DPO objective — Rafailov et al. 2023, Eq. 7:

        L_DPO = -E_{(x,y_w,y_l)} [
            log σ(
                β · [ log πθ(y_w|x)/πref(y_w|x)
                    - log πθ(y_l|x)/πref(y_l|x) ]
            )
        ]

    Implicit reward interpretation:
        r̂(x,y) = β · log [ πθ(y|x) / πref(y|x) ]

    The loss pushes the policy to assign a *higher implicit reward* to
    chosen over rejected.

    β controls the KL-divergence penalty strength:
      small β → policy can deviate far from the reference (aggressive)
      large β → policy stays close to the SFT checkpoint (conservative)

    Returns
    -------
    loss             — scalar (differentiable w.r.t. policy parameters)
    chosen_rewards   — [B] implicit rewards for chosen  (detached, for logging)
    rejected_rewards — [B] implicit rewards for rejected (detached, for logging)
    """
    # Log-ratio: log [ πθ(y|x) / πref(y|x) ] = implicit reward / β
    chosen_log_ratios = policy_chosen_logps - ref_chosen_logps  # [B]
    rejected_log_ratios = policy_rejected_logps - ref_rejected_logps  # [B]

    # Reward margin: how much more the policy prefers chosen over rejected
    # relative to the reference model's opinion.
    reward_margin = beta * (chosen_log_ratios - rejected_log_ratios)  # [B]

    # F.logsigmoid(x) = log σ(x) = -softplus(-x)   — numerically stable.
    # Manual log(sigmoid(x)) would underflow to -inf for large negative x.
    loss = -F.logsigmoid(reward_margin).mean()

    # Detach before returning: these are for logging only, not the loss graph.
    chosen_rewards = (beta * chosen_log_ratios).detach()  # [B]
    rejected_rewards = (beta * rejected_log_ratios).detach()  # [B]

    return loss, chosen_rewards, rejected_rewards


# ---------------------------------------------------------------------------
# 5. Training step
# ---------------------------------------------------------------------------


def training_step(
    batch: dict,
    policy_model: torch.nn.Module,
    ref_model: torch.nn.Module,
    beta: float = 0.1,
    pad_token_id: int = 0,
    average_log_prob: bool = True,
) -> Tuple[Tensor, dict]:
    """
    One DPO training step.

    Expected batch keys (all LongTensors of shape [B, L]):
        chosen_input_ids        — token IDs for prompt + chosen response
        chosen_attention_mask   — 1 for real tokens, 0 for padding
        chosen_labels           — -100 for prompt tokens, real IDs for response
        rejected_input_ids
        rejected_attention_mask
        rejected_labels

    policy_model  — πθ: trainable, receives gradients
    ref_model     — πref: frozen SFT checkpoint, MUST be in eval mode

    Returns
    -------
    loss    — scalar; call .backward() then optimizer.step()
    metrics — logging dict (all plain Python scalars)
    """
    # The reference model must be in eval mode so its forward pass is
    # deterministic. Active dropout would make log πref(y|x) stochastic,
    # adding noise to the log-ratios that is not part of the DPO signal.
    assert not ref_model.training, (
        "ref_model must be in eval mode. "
        "Call ref_model.eval() at initialisation."
    )

    device = next(policy_model.parameters()).device

    chosen_ids = batch["chosen_input_ids"].to(device)
    chosen_mask = batch["chosen_attention_mask"].to(device)
    chosen_lbls = batch["chosen_labels"].to(device)
    rejected_ids = batch["rejected_input_ids"].to(device)
    rejected_mask = batch["rejected_attention_mask"].to(device)
    rejected_lbls = batch["rejected_labels"].to(device)

    # --- Policy model (πθ): gradients flow through this pass ---
    policy_chosen_logps, policy_rejected_logps = concatenated_forward(
        policy_model,
        chosen_ids,
        chosen_mask,
        chosen_lbls,
        rejected_ids,
        rejected_mask,
        rejected_lbls,
        pad_token_id=pad_token_id,
        average_log_prob=average_log_prob,
    )

    # --- Reference model (πref): no computation graph ---
    with torch.no_grad():
        ref_chosen_logps, ref_rejected_logps = concatenated_forward(
            ref_model,
            chosen_ids,
            chosen_mask,
            chosen_lbls,
            rejected_ids,
            rejected_mask,
            rejected_lbls,
            pad_token_id=pad_token_id,
            average_log_prob=average_log_prob,
        )

    loss, chosen_rewards, rejected_rewards = dpo_loss(
        policy_chosen_logps,
        policy_rejected_logps,
        ref_chosen_logps,
        ref_rejected_logps,
        beta=beta,
    )

    metrics = {
        "loss": loss.item(),
        "reward_margin": (chosen_rewards - rejected_rewards).mean().item(),
        "chosen_reward_mean": chosen_rewards.mean().item(),
        "rejected_reward_mean": rejected_rewards.mean().item(),
        # Fraction of batch where chosen reward > rejected reward.
        # Should trend toward 1.0 as training progresses.
        "accuracy": (chosen_rewards > rejected_rewards).float().mean().item(),
        # Monitor for policy collapse: if logps/chosen or logps/rejected
        # drifts toward -inf, the policy is assigning near-zero probability
        # to all responses (reward hacking or mode collapse).
        "logps/chosen": policy_chosen_logps.detach().mean().item(),
        "logps/rejected": policy_rejected_logps.detach().mean().item(),
    }

    return loss, metrics
