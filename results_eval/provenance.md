# Results Provenance & Parameter Reference

## Experiment Overview

**Base model:** `Qwen/Qwen2.5-Coder-1.5B-Instruct`
**Training data:** `hh_train_3k.jsonl` — 3,000 preference pairs from Anthropic HH-RLHF
**Eval data:** `hh_eval_500.jsonl` — 500 held-out preference pairs
**Algorithm:** DPO (Direct Preference Optimization, offline)
**Fine-tuning method:** LoRA

### LoRA Config
| Param | Value |
|---|---|
| rank (r) | 16 |
| lora_alpha | 32 |
| lora_dropout | 0.0 |
| target_modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| trainable params | ~18.5M / 1.56B (1.18%) |

---

## Commands That Produced Each File

### Baseline files

**`results/baseline_hh.json`**
```bash
python -m eval.eval_hh \
  --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --eval_file hh_eval_500.jsonl \
  --out results/baseline_hh.json
```

**`results/baseline_humaneval.json`**
```bash
python -m eval.eval_humaneval \
  --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --out results/baseline_humaneval.json
```

---

### lr5e-7 run (too small — model did not move)

**Training:**
```bash
python -m dpo.train_trl \
  --output_dir qwen-coder-dpo-hh-lr5e7 \
  --lr 5e-7 \
  --epochs 2 \
  --beta 0.1 \
  --batch_size 4 \
  --grad_accum 8 \
  --max_length 768
# saved checkpoint to: qwen-coder-dpo-hh-lr5e7/
```

**`results/dpo_hh-lr5e7.json`**
```bash
python -m eval.eval_hh \
  --model ./qwen-coder-dpo-hh-lr5e7 \
  --eval_file hh_eval_500.jsonl \
  --out results/dpo_hh-lr5e7.json
```

**`results/dpo_hh_with_rm-lr5e7.json`**
```bash
python -m eval.eval_hh \
  --model ./qwen-coder-dpo-hh-lr5e7 \
  --eval_file hh_eval_500.jsonl \
  --with_rm \
  --out results/dpo_hh_with_rm-lr5e7.json
```

**`results/dpo_humaneval_rerun-lr5e7.json`**
```bash
python -m eval.eval_humaneval \
  --model ./qwen-coder-dpo-hh-lr5e7 \
  --out results/dpo_humaneval_rerun-lr5e7.json
```

**`results/dpo_margin_delta.json`**
```bash
python eval/eval_margin_delta.py \
  --trained_model ./qwen-coder-dpo-hh-lr5e7 \
  --eval_file hh_eval_500.jsonl \
  --out results/dpo_margin_delta.json
```

---

### lr2e-5 run (main experiment)

**Training:**
```bash
python -m dpo.train_trl \
  --output_dir qwen-coder-dpo-hh-lr2e5 \
  --lr 2e-5 \
  --epochs 2 \
  --beta 0.1 \
  --batch_size 4 \
  --grad_accum 8 \
  --max_length 768
# log saved to: output/dpo_lr2e5.log
# checkpoint saved to: qwen-coder-dpo-hh-lr2e5/
```

**`results/dpo_hh_lr2e5.json`**
```bash
python -m eval.eval_hh \
  --model ./qwen-coder-dpo-hh-lr2e5 \
  --eval_file hh_eval_500.jsonl \
  --out results/dpo_hh_lr2e5.json
```

**`results/dpo_humaneval_lr2e5.json`**
```bash
python -m eval.eval_humaneval \
  --model ./qwen-coder-dpo-hh-lr2e5 \
  --out results/dpo_humaneval_lr2e5.json
```

**`results/dpo_margin_delta_lr2e5.json`**
```bash
python eval/eval_margin_delta.py \
  --trained_model ./qwen-coder-dpo-hh-lr2e5 \
  --eval_file hh_eval_500.jsonl \
  --out results/dpo_margin_delta_lr2e5.json
```

---

## Parameter Glossary

### Eval result fields

| Field | File(s) | Meaning |
|---|---|---|
| `M1_preference_accuracy` | baseline_hh, dpo_hh-lr5e7 | Fraction of eval pairs where `logP(chosen) > logP(rejected)`. Binary: did the model assign higher probability to the preferred response? |
| `preference_accuracy` | dpo_hh_lr2e5, dpo_hh_with_rm-lr5e7 | Same metric, different key name used in later eval scripts. |
| `M3_mean_response_length` | baseline_hh, dpo_hh-lr5e7 | Average token count of model-generated responses on eval prompts. |
| `mean_response_length` | dpo_hh_lr2e5, dpo_hh_with_rm-lr5e7 | Same as above, newer key name. |
| `M4_humaneval_pass_at_1` | baseline_humaneval, dpo_humaneval_* | Fraction of HumanEval problems solved correctly with one generation attempt. Measures coding capability; should not regress after DPO. |
| `rm_score_mean` | dpo_hh_with_rm-lr5e7, dpo_hh_lr2e5 | Mean reward model score across eval pairs. Computed by the RM (separate from DPO objective). Negative = RM judges responses as below neutral. DPO does not optimize this directly so it can diverge. |
| `rm_score_median` | same | Median RM score; less sensitive to outliers than mean. |
| `n_problems` / `num_pairs` | all | Number of eval examples used. |

### Margin delta fields (`dpo_margin_delta*.json`)

| Field | Meaning |
|---|---|
| `base_preference_accuracy` | Fraction of pairs where base model has positive margin (logP_chosen > logP_rejected). |
| `trained_preference_accuracy` | Same for the DPO-trained model. |
| `base_margin` / `trained_margin` | Summary stats (mean, median, std, min, max) of per-pair log-prob margin: `logP(chosen|prompt) − logP(rejected|prompt)`, summed over response tokens. High variance is expected because response lengths vary enormously. |
| `delta_trained_minus_base` | Per-pair difference: `trained_margin − base_margin`. Positive = DPO moved the model in the right direction for that pair. |
| `fraction_delta_positive` | Fraction of 500 pairs where delta > 0. Random = 0.5; above 0.5 is a positive signal. |

### Training log fields (from `output/dpo_lr2e5.log`)

| Field | Meaning |
|---|---|
| `loss` | DPO loss: cross-entropy over the sigmoid of (β × log-ratio difference). Converging toward ~0.55 by epoch 2. |
| `grad_norm` | L2 norm of gradients. Stable ~2.0–2.8 throughout; no explosion. |
| `rewards/chosen` | Mean implicit reward for chosen responses: `β × (logπ(chosen) − logπ_ref(chosen))`. Should trend positive. |
| `rewards/rejected` | Same for rejected. Should trend negative. |
| `rewards/accuracies` | Fraction of training batch where chosen reward > rejected reward. Rises from ~0.49 to ~0.82 over training — the model is learning on train set. |
| `rewards/margins` | Mean `reward_chosen − reward_rejected` per batch. Increases from ~0.001 to ~0.41 — healthy signal on train data. |
| `logps/chosen` / `logps/rejected` | Mean sum log-prob of chosen/rejected responses per batch token. |
| `eval_rewards/accuracies` | Same as rewards/accuracies but on held-out eval set. 0.548 at epoch 1, 0.566 at epoch 2 — modest generalization. |
| `eval_rewards/margins` | Eval reward margin. 0.023 at epoch 1, 0.088 at epoch 2. Positive but small. |
