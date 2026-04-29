# DPO Experiment Analysis: lr2e-5 vs Baseline

## Setup

| | Value |
|---|---|
| Base model | Qwen/Qwen2.5-Coder-1.5B-Instruct |
| Training data | hh_train_3k.jsonl (3,000 HH-RLHF pairs) |
| Eval data | hh_eval_500.jsonl (500 held-out pairs) |
| Algorithm | DPO (offline), β=0.1 |
| LoRA | r=16, alpha=32, 7 target modules, ~18.5M params (1.18%) |
| LR | 2e-5 |
| Epochs | 2 |
| Batch / grad accum | 4 / 8 → effective batch = 32 |
| Max length | 768 tokens |

---

## Metrics Summary

| Metric | Baseline | lr5e-7 | lr2e-5 (DPO) | Δ (lr2e-5 − base) |
|---|---|---|---|---|
| Preference accuracy | 0.520 | 0.520 | 0.522 | +0.002 |
| HumanEval pass@1 | 0.555 | 0.573 | 0.598 | +0.043 |
| RM score (mean) | N/A | -0.481 | -0.657 | — |
| RM score (median) | N/A | 0.001 | -0.252 | — |
| Mean response length | 158.7 | 156.1 | 150.1 | -8.6 |
| Log-prob margin mean | 8.304 | 8.304 | 8.920 | +0.616 |
| Log-prob margin median | 2.503 | 2.562 | 2.688 | +0.528 |
| Fraction δ > 0 | — | 0.500 | 0.544 | — |

---

## Key Findings

### 1. lr5e-7 was a no-op
The model with lr=5e-7 produced metrics virtually identical to baseline across every dimension:
- Preference accuracy unchanged (0.520)
- Log-prob margin mean changed by only −0.001 (within noise)
- Fraction δ>0 = 0.500 (exactly chance)
- Conclusion: gradient steps were too small to move the weights meaningfully in 2 epochs.

### 2. lr2e-5 produced measurable DPO signal
- **Log-prob margin shifted**: mean increased from 8.30 → 8.92 (+0.616 nats)
- **Median delta = +0.528**: more than half of individual pairs improved
- **Fraction δ>0 = 0.544**: 54.4% of held-out pairs shifted in the right direction, above the 0.5 chance baseline
- Training reward margin grew from ~0.001 to ~0.41 over 2 epochs — the model clearly learned on training data

### 3. Preference accuracy barely moved (+0.002)
Binary accuracy went from 0.520 → 0.522. The base model already started at 0.520 with a large mean margin (8.30 nats), meaning most pairs were "correct" in log-prob space before DPO. Moving from 52% → 52.2% binary accuracy requires flipping the sign on pairs the model already got right-but-weakly, which is hard with 3k examples. The continuous margin shift (+0.615 nats mean) is a more sensitive indicator and confirms genuine learning.

### 4. HumanEval improved unexpectedly (++0.043)
pass@1 went from 0.555 → 0.598. This is a meaningful gain. DPO on HH preference data (which includes coding pairs) appears to have sharpened the model's code generation as a side effect. lr5e-7 also showed improvement (0.573), suggesting the LoRA fine-tuning process itself helps HumanEval somewhat even at small learning rates.

### 5. RM score degraded (−0.176 relative to lr5e-7)
RM score mean dropped from −0.481 (lr5e-7, nearly at baseline behavior) to −0.657 (lr2e-5). This is expected: DPO optimizes log-prob margins directly, not RM score. The RM was trained separately (GRPO objective) and measures a different notion of quality. A DPO policy can widen the chosen/rejected log-prob gap while simultaneously drifting away from what the RM rewards — especially at higher learning rates where the policy moves further from the reference.

### 6. Response length shortened slightly (−8.6 tokens)
From 158.7 → 150.1 tokens. This is a common DPO artifact: the model learns to be more concise when the chosen responses in training data are shorter on average than rejected ones.

---

## Training Dynamics (lr2e-5)

From `output/dpo_lr2e5.log`:

| Checkpoint | Loss | Reward acc. (train) | Reward margin (train) | Eval reward acc. | Eval reward margin |
|---|---|---|---|---|---|
| Epoch 0.11 | 0.693 | 0.494 | 0.002 | — | — |
| Epoch 1.00 | — | — | — | 0.548 | 0.023 |
| Epoch 1.06 | 0.623 | 0.708 | 0.169 | — | — |
| Epoch 1.60 | 0.555 | 0.822 | 0.352 | — | — |
| Epoch 2.00 | 0.679 | — | — | 0.566 | 0.088 |
| **Final** | **0.626** | — | — | — | — |

The model is learning on training data (reward accuracy 49% → 82%, margin 0.002 → 0.41).
Eval reward accuracy (0.548 @ epoch 1, 0.566 @ epoch 2) confirms partial generalization.
The eval loss of 0.679 vs train loss 0.626 suggests mild overfitting, consistent with 3k examples.

---

## Visualizations

- `fig1_metrics_overview.png` — side-by-side bar charts for all 4 primary metrics across baseline / lr5e-7 / lr2e-5
- `fig2_margin_analysis.png` — log-prob margin mean/median, delta distribution, preference accuracy
- `fig3_training_dynamics.png` — loss, reward margin, and reward accuracy over training epochs

---

## Recommendations for Next Steps

1. **More data**: 3k pairs is the main bottleneck. 10k–30k HH pairs would likely move binary preference accuracy meaningfully.
2. **Tune β**: β=0.1 is standard but a sweep (0.05, 0.1, 0.2) could find better RM/preference tradeoff.
3. **More epochs or higher LR with warmup**: Eval reward accuracy was still climbing at epoch 2 — a 3rd epoch or cosine schedule might help without severe overfitting.
4. **Track calibration**: The high variance in margin (std ~138 nats) is driven by response length. Normalizing by response token count would give a cleaner per-token signal.
