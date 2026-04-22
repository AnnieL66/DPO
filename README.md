# DPO Training for Code Generation

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
├── tests/
│   └── test_dpo.py          # 36-test pytest suite (no GPU required)
├── requirements.txt
└── setup_env.sh             # One-command environment setup
```

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

36 tests across 7 groups:

| Group | Count | What it covers |
|---|---|---|
| `TestBatchLogps` | 8 | shape, sign, prompt masking, sum scaling with length, dtype guard |
| `TestConcatenatedForward` | 3 | correctness vs separate calls, unequal-length sequences |
| `TestDPOLoss` | 7 | manual math, neutral = log(2), loss ordering, detach, β scaling |
| `TestTrainingStep` | 8 | forward, backward, gradient direction, ref model unchanged |
| `TestNumericalStability` | 4 | extreme logits (±1e6), uniform logits, logsigmoid stability |
| `TestDataUtils` | 5 | SHP score filters, HF Hub schema validation, raw data |
| `TestToyExample` | 1 | `run_all_checks()` end-to-end |

All 36 tests pass on CPU in under 2 seconds.

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

### Train on Stanford Human Preferences (SHP)

```bash
python -m dpo.train_trl \
  --dataset_name shp \
  --output_dir qwen-coder-dpo-shp \
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
| `--dataset_name` | _(none)_ | `hh`, `shp`, or any HF Hub dataset name with `prompt`/`chosen`/`rejected` columns. Omit to use the 3-example built-in sample data. |
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
tensorboard --logdir qwen-coder-dpo
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
- **Reference model** — same base checkpoint, no LoRA, all parameters frozen, always in eval mode.
- Both models share no weights at runtime. Two copies at bfloat16 ≈ 6 GB total VRAM.

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

To add a new named dataset like `hh` or `shp`, edit `dpo/data_utils.py`:

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
    elif name == "shp":
        return get_shp(split, **kwargs)
    elif name == "my_dataset":
        return get_my_dataset(split, **kwargs)
    ...
```

**Step 3 — Register the truncation mode** in `DATASET_TRUNCATION_MODE`:

```python
DATASET_TRUNCATION_MODE = {
    "hh":         "keep_end",    # multi-turn: preserve the most recent context
    "shp":        "keep_start",  # single question: preserve the question
    "my_dataset": "keep_start",
}
```

**Step 4 — Train:**

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
