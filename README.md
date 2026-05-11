# DPO Training for Code Generation
Github link: https://github.com/AnnieL66/DPO
**Base model:** `Qwen/Qwen2.5-Coder-1.5B-Instruct`  
**Comparison:** This repo implements the DPO arm of a two-method RLHF study (Reward Model + GRPO vs. DPO) for a CS 5788 course project.

---

## Repository Structure

```
DPO/
├── dpo/
│   ├── __init__.py          # Package metadata
│   ├── dpo_loss.py          # Core DPO loss mathematics
│   ├── model_utils.py       # Policy and reference model loading (LoRA)
│   ├── data_utils.py        # Dataset loading and preprocessing
│   ├── train_trl.py         # TRL DPOTrainer entry point (recommended)
│   └── toy_example.py       # CPU-only loss verification script
├── eval/
│   ├── eval_hh.py           # Preference accuracy and RM score on HH eval pairs
│   ├── eval_humaneval.py    # HumanEval pass@1 coding benchmark
│   └── eval_margin_delta.py # Log-prob margin comparison: base vs trained model
├── results/                 # Raw JSON outputs from eval scripts
├── results_eval/            # Analysis, visualizations, and documentation (see below)
├── tests/
│   └── test_dpo.py          # 34-test pytest suite (no GPU required)
├── requirements.txt
└── setup_env.sh             # One-command environment setup
```

---

## Evaluation Results (`results_eval/`)

Pre-computed analysis comparing the DPO-trained model (`lr2e-5`) against the base model.

| File | Description |
|---|---|
| `provenance.md` | Every command that produced each file in `results/`, plus a full glossary of all JSON fields and training log metrics |
| `analysis.md` | Comprehensive analysis: metrics table, key findings, training dynamics, and recommendations for next steps |
| `margin_explainer.md` | Plain-English explanation of log-prob margin and delta, with a worked example and full stats table |
| `fig1_metrics_overview.png` | 2×2 bar chart — preference accuracy, HumanEval pass@1, RM score, and response length: baseline vs lr2e-5 |
| `fig2_margin_analysis.png` | Log-prob margin mean/median, delta distribution, and fraction of pairs that improved |
| `fig3_training_dynamics.png` | Loss, reward margin, and reward accuracy over 2 training epochs with eval checkpoints |
| `generate_analysis.py` | Script that regenerates `fig1–3` and `analysis.md` from the raw JSON files in `results/` |

### Quick summary of results

| Metric | Baseline | DPO lr2e-5 | Δ |
|---|---|---|---|
| Preference accuracy | 0.520 | 0.522 | +0.002 |
| HumanEval pass@1 | 0.555 | 0.598 | +0.043 |
| RM score (mean) | N/A | −0.657 | — |
| Log-prob margin (mean) | 8.30 | 8.92 | +0.615 |
| Fraction of pairs improved (δ > 0) | — | 54.4% | — |

DPO at lr=2e-5 produced a measurable shift in log-prob preference margins (+0.615 nats mean, 54.4% of pairs improved) and a notable HumanEval gain (+4.3 pp). Binary preference accuracy barely moved because the base model was already weakly calibrated before DPO. See `results_eval/analysis.md` for full details.

---

## Setup

### Prerequisites

- Python 3.10+
- CUDA-capable GPU with at least 24 GB VRAM (e.g., NVIDIA L4/A10/A100)  
  For smaller GPUs, use `--use_4bit` (see Training section)
- Internet access to download the model and datasets from HuggingFace

### Install

```bash
bash setup_env.sh
source venv/bin/activate
```

`setup_env.sh` creates a virtual environment and installs all dependencies from `requirements.txt`.

### Manual install

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Verification

### Toy example (CPU only, no model download)

Run this first to confirm the loss math is correct before touching a GPU:

```bash
python -m dpo.toy_example
```

This runs three checks:

1. **Hand-crafted log-probs** — verifies loss ordering: correct policy < neutral policy < wrong policy, and that the neutral case equals log(2) ≈ 0.693
2. **Tiny random model** — checks output shapes and sign constraints on `compute_log_probs`
3. **Gradient step** — asserts one backward pass reduces the loss

### Test suite (CPU only, no model download)

```bash
pytest tests/test_dpo.py -v
```

34 tests across 7 groups:

| Group | Count | What it covers |
|---|---|---|
| `TestBatchLogps` | 8 | shape, sign, prompt masking, sum scaling with length, dtype guard |
| `TestConcatenatedForward` | 3 | correctness vs separate calls, unequal-length sequences |
| `TestDPOLoss` | 7 | manual math, neutral = log(2), loss ordering, detach, β scaling |
| `TestTrainingStep` | 8 | forward, backward, gradient direction, ref model unchanged |
| `TestNumericalStability` | 4 | extreme logits (±1e6), uniform logits, logsigmoid stability |
| `TestDataUtils` | 3 | HF Hub schema validation, raw data, sample data |
| `TestToyExample` | 1 | `run_all_checks()` end-to-end |

All 34 tests pass on CPU in under 2 seconds.

---

## Training

### Quick start with built-in sample data

Runs the full pipeline on three hard-coded examples to verify the setup end-to-end:

```bash
python -m dpo.train_trl
```

### Train on Anthropic Helpful-Harmless (HH)

```bash
python -m dpo.train_trl \
  --dataset_name hh \
  --output_dir qwen-coder-dpo-hh \
  --epochs 3
```

### Train on a HuggingFace Hub dataset

Pass any HuggingFace Hub dataset name that already has `prompt`, `chosen`, and `rejected` string columns:

```bash
python -m dpo.train_trl \
  --dataset_name owner/my-dpo-dataset \
  --output_dir qwen-coder-dpo-custom \
  --epochs 3
```

If the dataset uses a different eval split name than `test`, specify it:

```bash
python -m dpo.train_trl \
  --dataset_name owner/my-dpo-dataset \
  --eval_split validation \
  --output_dir qwen-coder-dpo-custom
```

### Train on a smaller GPU (4-bit quantization)

```bash
python -m dpo.train_trl \
  --dataset_name hh \
  --output_dir qwen-coder-dpo-4bit \
  --use_4bit
```

### All command-line options

| Flag | Default | Description |
|---|---|---|
| `--model_name` | `Qwen/Qwen2.5-Coder-1.5B-Instruct` | HuggingFace model ID |
| `--output_dir` | `qwen-coder-dpo` | Directory for checkpoints and final model |
| `--dataset_name` | _(none)_ | `hh`, `hh_local`, or any HF Hub dataset name with `prompt`/`chosen`/`rejected` columns. Omit to use the 3-example built-in sample data. |
| `--eval_split` | `test` | Eval split name. Some datasets use `validation` instead of `test`. |
| `--beta` | `0.1` | KL-divergence penalty strength |
| `--lr` | `5e-7` | Learning rate |
| `--epochs` | `3` | Number of training epochs |
| `--batch_size` | `4` | Per-device batch size |
| `--grad_accum` | `8` | Gradient accumulation steps (effective batch = 32) |
| `--max_length` | `768` | Total token length (prompt + response) |
| `--max_prompt_length` | `256` | Maximum prompt length in tokens |
| `--use_4bit` | `False` | Enable 4-bit (NF4) quantization for GPUs under 16 GB |

### Monitor training

Logs are written to TensorBoard every 10 steps:

```bash
tensorboard --logdir runs/qwen-coder-dpo
```

Key metrics to watch:

| Metric | Healthy trend |
|---|---|
| `loss` | Decreasing |
| `rewards/chosen` | Increasing (policy assigns higher implicit reward to preferred responses) |
| `rewards/rejected` | Decreasing |
| `rewards/accuracies` | Trending toward 1.0 (chosen reward > rejected reward for most examples) |
| `logps/chosen` | Should not drift toward −∞ (policy collapse) |
| `logps/rejected` | Should not drift toward −∞ (policy collapse) |

---

## How DPO Training Works

### Algorithm

DPO optimizes the policy directly from preference pairs without training a separate reward model. For each `(prompt x, chosen y_w, rejected y_l)` triplet, the loss is:

```
L = -log σ( β × [ (log πθ(y_w|x) − log πref(y_w|x)) − (log πθ(y_l|x) − log πref(y_l|x)) ] )
```

- `πθ` — policy model (LoRA-adapted, trainable)
- `πref` — reference model (frozen copy of the same SFT checkpoint)
- `β = 0.1` — controls the strength of the KL-divergence penalty to the reference

The quantity `β · log(πθ(y|x) / πref(y|x))` is the *implicit reward* the policy assigns to response `y`. The loss pushes this reward to be higher for `y_w` than for `y_l`.

At the start of training when `πθ = πref`, every log-ratio is 0, the reward margin is 0, `σ(0) = 0.5`, and the loss equals `log(2) ≈ 0.693`. This is verified by `TestDPOLoss::test_neutral_case_equals_log2`.

### Log-probability computation

Sequence-level log-probabilities are computed as the **sum of token log-probs** (not the mean), matching Eq. 7 of Rafailov et al. 2023:

```
log P(y|x) = Σ log πθ(y_t | x, y_1..y_{t-1})
```

Using the sum is the vanilla DPO formulation and keeps the comparison with GRPO on the same footing. The `_batch_logps` function accepts an `average_log_prob` flag (default `False`) if length-normalised behaviour is needed.

### Model architecture

- **Policy model** — Qwen2.5-Coder-1.5B-Instruct with LoRA adapters (r=16, α=32) on all attention and MLP projection layers. Only ~4.5M of 1.5B parameters are trainable (~0.3%).
- **Reference model** — same base checkpoint, accessed via `model.disable_adapter()` (TRL's `ref_model=None` path), so only one copy of the base weights is loaded. This cuts VRAM roughly in half compared to two separate model objects.

### Efficiency

- **Concatenated forward pass** — chosen and rejected are stacked into a `[2B, L]` tensor so each model runs one forward pass per step instead of two.
- **Float32 log-softmax** — Qwen has a 152k-token vocabulary. Log-softmax is computed in float32 to avoid precision loss before being cast back to bfloat16.
- **Dropout disabled** — both models have all dropout layers zeroed so policy and reference forward passes are deterministic.

---

## Adding a New Dataset

### Option 1: Use a HuggingFace Hub dataset directly

If your dataset is on the Hub and already has `prompt`, `chosen`, and `rejected` string columns:

```bash
python -m dpo.train_trl --dataset_name owner/my-dataset
```

The loader validates that these three columns exist and raises a clear error if they are missing.

### Option 2: Add a built-in loader

To add a new named dataset like `hh`, edit `dpo/data_utils.py`:

**Step 1 — Write a loader** that returns a list of dicts with keys `prompt`, `chosen`, `rejected` (all plain strings):

```python
def get_my_dataset(split: str = "train", silent: bool = False) -> List[Dict]:
    from datasets import load_dataset
    raw = load_dataset("owner/my-dataset", split=split)
    data = []
    for row in raw:
        data.append({
            "prompt":   row["question"],
            "chosen":   row["good_answer"],
            "rejected": row["bad_answer"],
        })
    return data
```

**Step 2 — Register it in `load_dataset_by_name()`:**

```python
def load_dataset_by_name(name: str, split: str, **kwargs):
    if name == "hh":
        return get_hh(split, **kwargs)
    elif name == "hh_local":
        return get_hh_local(split, **kwargs)
    elif name == "my_dataset":
        return get_my_dataset(split, **kwargs)
    ...
```

**Step 3 — Train:**

```bash
python -m dpo.train_trl --dataset_name my_dataset --output_dir my-dataset-dpo
```

### Option 3: Pass data directly in code

```python
from dpo.data_utils import build_trl_dataset

raw_data = [
    {
        "prompt":   "Write a function to reverse a string.",
        "chosen":   "def reverse(s):\n    return s[::-1]",
        "rejected": "def reverse(s):\n    return s",
    },
]

dataset = build_trl_dataset(raw_data=raw_data)
```

---

## Key Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| β (beta) | 0.1 | Moderate KL penalty; keeps policy close to reference |
| Learning rate | 5e-7 | Much lower than SFT to avoid catastrophic forgetting |
| Effective batch | 32 | 4 per device × 8 gradient accumulation steps |
| Epochs | 3 | On a static offline dataset |
| Max sequence length | 768 | Covers most code prompts + short solutions |
| LoRA rank | 16 | Balances expressiveness vs. parameter count |

---

## Dependencies

| Package | Version |
|---|---|
| torch | ≥ 2.1.0 |
| transformers | ≥ 4.40.0 |
| trl | ≥ 0.9.0 |
| peft | ≥ 0.10.0 |
| datasets | ≥ 2.18.0 |
| accelerate | ≥ 0.28.0 |
| bitsandbytes | ≥ 0.43.0 |
| tensorboard | ≥ 2.16.0 |
