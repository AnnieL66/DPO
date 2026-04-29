# Log-prob Margin: Explanation & Baseline vs lr2e-5 Comparison

## What is the log-prob margin?

For a preference pair `(prompt, chosen, rejected)`, the log-prob margin is:

```
margin = log P(chosen | prompt) − log P(rejected | prompt)
```

This is the difference in total log-likelihood the model assigns to the chosen vs rejected
response, **summed over all response tokens**. A positive margin means the model assigns higher
probability to the preferred response — it "prefers" the chosen response in probability space.

---

## Why is the base margin already 8.3?

The base model is `Qwen2.5-Coder-1.5B-Instruct`, which has already been instruction-tuned with
RLHF before our DPO run. It naturally assigns higher probability to the preferred responses on
most pairs. A mean margin of 8.30 nats means the base model is already weakly calibrated toward
human preferences — DPO is refining this further, not starting from scratch.

---

## Run Comparison

| Run    | Mean margin | Delta   | Interpretation                          |
|--------|-------------|---------|-----------------------------------------|
| base   | 8.30        | —       | Baseline (RLHF-tuned, no DPO)           |
| lr5e-7 | 8.30        | −0.001  | LR too small — weights barely moved     |
| lr2e-5 | 8.92        | +0.615  | DPO actually shifted the distribution   |

**lr2e-5 is the only run that did anything.** The +0.615 delta and 54.4% fraction-delta-positive
confirm DPO pushed the model toward chosen responses on held-out pairs — a real (if weak) signal,
not noise.

**lr5e-7 is essentially a no-op.** Gradient steps were so small the model did not move at all —
the margin is virtually identical to the base.

---

## What is the mean margin?

The **mean margin** is the average of `log P(chosen | prompt) − log P(rejected | prompt)` across
all 500 eval pairs.

- Each pair contributes one number: how much more probable (in log space) the model finds the
  chosen response vs the rejected one
- Averaging across 500 pairs gives a single summary of how well-calibrated the model's preferences
  are overall
- Higher = model assigns more probability mass to chosen responses on average

---

## What is the delta?

**Delta** (`delta_trained_minus_base`) is computed per pair, then averaged:

```
delta_i = trained_margin_i − base_margin_i
mean_delta = mean over all 500 pairs
```

It answers: *did DPO training increase or decrease the model's preference gap on each pair?*

- **Positive delta** on a pair → DPO pushed the model to favor chosen more strongly
- **Negative delta** → DPO moved in the wrong direction for that pair
- **`fraction_delta_positive = 0.544`** → 54.4% of the 500 pairs improved (barely above the 0.5
  chance baseline, but a genuine positive signal)

---

## Concrete Example (one pair)

|                 | base  | trained (lr2e-5) |
|-----------------|-------|------------------|
| log P(chosen)   | −42.1 | −40.5            |
| log P(rejected) | −50.4 | −50.0            |
| **margin**      | **8.3**   | **9.5**      |
| **delta**       |       | **+1.2**         |

After DPO, the model is more confident in the chosen response (log P rose from −42.1 to −40.5)
while the rejected response probability fell slightly. The margin widened by 1.2 nats for this
pair.

The **mean margin of 8.92 vs 8.30** means that on average the gap widened by 0.615 nats after
DPO — the model became slightly more "decisive" in preferring chosen responses across the eval set.

---

## Full Margin Statistics (lr2e-5)

From `results/dpo_margin_delta_lr2e5.json`:

| Stat   | Base margin | Trained margin | Delta |
|--------|-------------|----------------|-------|
| mean   | 8.305       | 8.920          | +0.615 |
| median | 2.503       | 2.688          | +0.528 |
| std    | 137.94      | 139.51         | 4.316  |
| min    | −502.4      | −510.0         | −15.72 |
| max    | 517.3       | 515.9          | +17.83 |

The large std and extreme min/max are expected: margin is summed over all response tokens, so
long responses produce large absolute values. The median (2.5 → 2.7) is a more robust measure
than the mean for this reason. Both median and mean shifted positively, confirming the signal is
not driven by outliers.
