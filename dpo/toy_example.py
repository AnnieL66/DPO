"""
toy_example.py
--------------
Self-contained verification of the DPO loss — no GPU or model download needed.

Three checks
------------
Part 1: dpo_loss with hand-crafted log-probs
  Case A: policy prefers chosen  → LOW loss
  Case B: policy prefers rejected (wrong) → HIGH loss
  Case C: policy == reference    → loss = log(2) ≈ 0.693

Part 2: compute_log_probs shape/sign checks with a tiny random model

Part 3: one gradient step lowers the loss

Run:
    python -m dpo.toy_example
"""

import math
import torch
import torch.nn as nn

from .dpo_loss import dpo_loss, compute_log_probs

BETA = 0.1


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Minimal causal LM stub used for Parts 2 and 3.
# No self-attention — just Embedding → Linear.  Enough to produce
# correctly shaped logits for testing _batch_logps and the gradient path.
# ---------------------------------------------------------------------------

class TinyLM(nn.Module):
    VOCAB = 12

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(self.VOCAB, 8)
        self.proj  = nn.Linear(8, self.VOCAB)

    def forward(self, input_ids, attention_mask=None):
        logits = self.proj(self.embed(input_ids))

        class _Out:
            pass

        out = _Out()
        out.logits = logits
        return out


# ---------------------------------------------------------------------------
# Part 1: dpo_loss with hand-crafted log-probabilities
# ---------------------------------------------------------------------------

def check_dpo_loss_cases() -> None:
    section("Part 1: dpo_loss with hand-crafted log-probabilities")

    print("""
Notation
--------
  pi_theta = policy model (being trained)
  pi_ref   = reference model (frozen SFT checkpoint)

  log_ratio_chosen   = log pi_theta(y_chosen | x) - log pi_ref(y_chosen | x)
  log_ratio_rejected = log pi_theta(y_rejected | x) - log pi_ref(y_rejected | x)

  DPO loss = -log sigma( beta * (log_ratio_chosen - log_ratio_rejected) )
""")

    ref_chosen   = torch.tensor([-15.0])
    ref_rejected = torch.tensor([-20.0])

    # Case A — policy correctly prefers chosen
    print("Case A — Policy CORRECTLY prefers chosen over rejected")
    policy_chosen_A   = torch.tensor([-10.0])
    policy_rejected_A = torch.tensor([-25.0])
    loss_A, ch_rew_A, rej_rew_A = dpo_loss(
        policy_chosen_A, policy_rejected_A,
        ref_chosen, ref_rejected, beta=BETA,
    )
    margin_A = (ch_rew_A - rej_rew_A).item()
    print(f"  policy_chosen_logp   = {policy_chosen_A.item():.1f}")
    print(f"  policy_rejected_logp = {policy_rejected_A.item():.1f}")
    print(f"  reward margin        = {margin_A:.4f}")
    print(f"  DPO loss             = {loss_A.item():.4f}  <- should be LOW")

    # Case B — policy incorrectly prefers rejected
    print("\nCase B — Policy INCORRECTLY prefers rejected over chosen")
    policy_chosen_B   = torch.tensor([-25.0])
    policy_rejected_B = torch.tensor([-10.0])
    loss_B, ch_rew_B, rej_rew_B = dpo_loss(
        policy_chosen_B, policy_rejected_B,
        ref_chosen, ref_rejected, beta=BETA,
    )
    margin_B = (ch_rew_B - rej_rew_B).item()
    print(f"  policy_chosen_logp   = {policy_chosen_B.item():.1f}")
    print(f"  policy_rejected_logp = {policy_rejected_B.item():.1f}")
    print(f"  reward margin        = {margin_B:.4f}")
    print(f"  DPO loss             = {loss_B.item():.4f}  <- should be HIGH")

    # Case C — policy identical to reference (start of training)
    print("\nCase C — Policy IDENTICAL to reference (start of training)")
    loss_C, _, _ = dpo_loss(
        ref_chosen.clone(), ref_rejected.clone(),
        ref_chosen, ref_rejected, beta=BETA,
    )
    print(
        f"  DPO loss = {loss_C.item():.4f}"
        f"  <- should be log(2) = {math.log(2):.4f}"
    )

    print("\nSummary")
    print(f"  Loss A (correct order):  {loss_A.item():.4f}")
    print(f"  Loss C (neutral):        {loss_C.item():.4f}")
    print(f"  Loss B (reversed order): {loss_B.item():.4f}")
    assert loss_A.item() < loss_C.item() < loss_B.item(), (
        "Loss ordering violated — expected loss_A < loss_C < loss_B."
    )
    print("  Ordering check PASSED: loss_A < loss_C < loss_B")


# ---------------------------------------------------------------------------
# Part 2: compute_log_probs shape/sign checks
# ---------------------------------------------------------------------------

def check_compute_log_probs() -> None:
    section("Part 2: compute_log_probs with a tiny random Transformer")

    torch.manual_seed(42)
    tiny_model = TinyLM()

    B, L = 2, 5
    input_ids      = torch.randint(0, TinyLM.VOCAB, (B, L))
    attention_mask = torch.ones(B, L, dtype=torch.long)
    labels         = input_ids.clone()
    labels[:, :2]  = -100   # mask first 2 positions as prompt

    logps = compute_log_probs(tiny_model, input_ids, attention_mask, labels)

    print(f"\n  input shape:     {tuple(input_ids.shape)}")
    print(f"  labels shape:    {tuple(labels.shape)}  (cols 0-1 = -100)")
    print(f"  log_probs shape: {tuple(logps.shape)}  (one scalar per sequence)")
    print(f"  log_probs:       {logps.tolist()}")

    assert logps.shape == (B,), f"Expected ({B},), got {logps.shape}"
    assert (logps <= 0).all(), "Log-probs must be <= 0"
    print("  Shape and sign checks PASSED")


# ---------------------------------------------------------------------------
# Part 3: gradient step lowers loss
# ---------------------------------------------------------------------------

def check_gradient_step() -> None:
    section("Part 3: Gradient step drives loss in the right direction")

    torch.manual_seed(0)
    tiny_policy = TinyLM()
    tiny_ref    = TinyLM()
    for p in tiny_ref.parameters():
        p.requires_grad = False
    tiny_ref.eval()

    opt = torch.optim.AdamW(tiny_policy.parameters(), lr=1e-3)

    B2, L2 = 1, 6
    chosen_ids      = torch.randint(0, TinyLM.VOCAB, (B2, L2))
    rejected_ids    = torch.randint(0, TinyLM.VOCAB, (B2, L2))
    mask            = torch.ones(B2, L2, dtype=torch.long)
    chosen_labels   = chosen_ids.clone()
    chosen_labels[:, :2]   = -100
    rejected_labels = rejected_ids.clone()
    rejected_labels[:, :2] = -100

    def forward_loss() -> torch.Tensor:
        pol_ch  = compute_log_probs(tiny_policy, chosen_ids,   mask, chosen_labels)
        pol_rej = compute_log_probs(tiny_policy, rejected_ids, mask, rejected_labels)
        with torch.no_grad():
            ref_ch  = compute_log_probs(tiny_ref, chosen_ids,   mask, chosen_labels)
            ref_rej = compute_log_probs(tiny_ref, rejected_ids, mask, rejected_labels)
        loss, _, _ = dpo_loss(pol_ch, pol_rej, ref_ch, ref_rej, beta=BETA)
        return loss

    loss_before = forward_loss()
    print(f"\n  Loss BEFORE gradient step: {loss_before.item():.6f}")

    opt.zero_grad()
    loss_before.backward()
    opt.step()

    with torch.no_grad():
        loss_after = forward_loss()

    delta = loss_after.item() - loss_before.item()
    print(f"  Loss AFTER  gradient step: {loss_after.item():.6f}")
    print(f"  delta = {delta:+.6f}  (negative means training is working)")

    assert loss_after.item() < loss_before.item(), (
        f"Loss did not decrease after gradient step: "
        f"{loss_before.item():.6f} -> {loss_after.item():.6f}"
    )
    print("  Gradient direction check PASSED")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_all_checks() -> None:
    check_dpo_loss_cases()
    check_compute_log_probs()
    check_gradient_step()
    print("\n" + "=" * 60)
    print("  All toy example checks completed.")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_all_checks()
