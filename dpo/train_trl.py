"""
train_trl.py
------------
TRL DPOTrainer entry-point for code-generation DPO.

Why TRL?
--------
TRL's DPOTrainer handles:
  - tokenisation of conversational datasets (chat templates)
  - reference model management (auto-freezing via disable_adapter())
  - per-token log-prob computation with prompt masking
  - gradient accumulation, mixed precision, logging

We configure it to match the exact hyperparameters from the project
proposal and the LoRA config from Section 2.3.

Usage
-----
    # Smoke test with built-in 3-example sample data:
    python -m dpo.train_trl

    # Load from the deterministic 3k JSONL split (recommended for this project):
    python -m dpo.train_trl --train_file hh_train_3k.jsonl

    # Named built-in datasets (hh or shp) — uses custom prompt extraction:
    python -m dpo.train_trl --dataset_name hh
    python -m dpo.train_trl --dataset_name shp

    # Any HuggingFace Hub dataset that already has prompt/chosen/rejected cols:
    python -m dpo.train_trl --dataset_name my-org/my-dpo-dataset

    # Full options:
    python -m dpo.train_trl \\
        --train_file hh_train_3k.jsonl \\
        --model_name Qwen/Qwen2.5-Coder-1.5B-Instruct \\
        --output_dir qwen-coder-dpo
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

from .data_utils import build_trl_dataset, SAMPLE_PREFERENCE_DATA
from .model_utils import LORA_CONFIG, load_tokenizer

BUILTIN_DATASETS = ("hh", "hh_local", "shp")


def parse_args():
    p = argparse.ArgumentParser(
        description="TRL DPO training for code generation"
    )
    p.add_argument(
        "--model_name", default="Qwen/Qwen2.5-Coder-1.5B-Instruct"
    )
    p.add_argument("--output_dir", default="qwen-coder-dpo")
    p.add_argument(
        "--train_file",
        default=None,
        help=(
            "Path to a local JSONL file with 'prompt', 'chosen', 'rejected' "
            "columns (e.g. hh_train_3k.jsonl from prepare_hh_split.py). "
            "Takes priority over --dataset_name when both are given."
        ),
    )
    p.add_argument(
        "--eval_file",
        default=None,
        help=(
            "Path to a local JSONL file for evaluation "
            "(e.g. hh_eval_500.jsonl). Used when --train_file is given."
        ),
    )
    p.add_argument(
        "--dataset_name",
        default=None,
        help=(
            "Dataset to train on. Use 'hh' or 'shp' for the built-in loaders, "
            "'hh_local' to use --train_file/--eval_file local JSONL files, "
            "or a HuggingFace Hub dataset name with 'prompt'/'chosen'/'rejected' "
            "columns. Omit to run a smoke test with the 3-example built-in data."
        ),
    )
    p.add_argument(
        "--eval_split",
        default="test",
        help=(
            "Name of the evaluation split to load. Most datasets use 'test'; "
            "some use 'validation'. If the split does not exist, evaluation "
            "is skipped with a warning."
        ),
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    p.add_argument("--beta",              type=float, default=0.1)
    p.add_argument("--lr",               type=float, default=5e-7)
    p.add_argument("--epochs",           type=int,   default=3)
    p.add_argument("--batch_size",       type=int,   default=4)
    p.add_argument("--grad_accum",       type=int,   default=8)
    p.add_argument("--max_length",       type=int,   default=768)
    p.add_argument("--max_prompt_length", type=int,  default=256,
                   help="Maximum prompt length in tokens (default 256)")
    p.add_argument(
        "--use_4bit",
        action="store_true",
        help="Load base model in 4-bit (for GPUs with less than 16 GB VRAM)",
    )
    return p.parse_args()


def build_model_for_trl(
    model_name: str,
    use_bf16: bool = True,
    use_4bit: bool = False,
):
    """
    Load the base model for TRL training.

    TRL's DPOTrainer creates its own frozen reference model internally when
    you pass a PEFT model: it calls model.disable_adapter() to run the
    reference forward pass.  This means we only load the base weights once —
    TRL handles the dual-model logic automatically, saving memory.
    """
    torch_dtype = torch.bfloat16 if use_bf16 else torch.float32

    if use_4bit:
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_cfg,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )

    return model


def _load_split(dataset_name: str, split: str):
    """
    Load one split of a dataset, routing to the correct loader.

    If dataset_name is a built-in name ("hh" or "shp"), the custom loader
    is used, which extracts the prompt correctly from the raw conversation
    format.  Passing "hh" directly as an HF Hub name would fail because
    the raw Anthropic/hh-rlhf dataset does not have a "prompt" column.

    For any other string, it is treated as an HF Hub dataset name and must
    already contain "prompt", "chosen", and "rejected" columns.

    Returns None (with a warning) if the split does not exist, so the
    caller can gracefully skip evaluation.
    """
    try:
        if dataset_name in BUILTIN_DATASETS:
            return build_trl_dataset(dataset_name=dataset_name, split=split)
        else:
            return build_trl_dataset(hf_dataset_name=dataset_name, split=split)
    except Exception as exc:
        print(f"  Warning: could not load split '{split}' — {exc}")
        return None


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    print("Loading dataset ...")
    if args.train_file or args.dataset_name == "hh_local":
        if args.train_file is None:
            raise RuntimeError(
                "--dataset_name hh_local requires --train_file to be set."
            )
        # Route through data_utils.get_hh_local so the loader lives in one place.
        train_data = build_trl_dataset(
            dataset_name="hh_local", split="train", filepath=args.train_file
        )
        train_dataset = train_data
        if args.eval_file:
            eval_dataset = build_trl_dataset(
                dataset_name="hh_local", split="eval", filepath=args.eval_file
            )
        else:
            eval_dataset = None
    elif args.dataset_name:
        train_dataset = _load_split(args.dataset_name, "train")
        eval_dataset  = _load_split(args.dataset_name, args.eval_split)
        if train_dataset is None:
            raise RuntimeError(
                f"Could not load training split for '{args.dataset_name}'."
            )
    else:
        print(
            "  [INFO] No --train_file or --dataset_name given. "
            "Using built-in 3-sample toy data.\n"
            "         This is a smoke test only; use a real dataset for training."
        )
        train_dataset = build_trl_dataset(raw_data=SAMPLE_PREFERENCE_DATA)
        eval_dataset  = None

    print(f"  Train examples: {len(train_dataset)}")
    if eval_dataset is not None:
        print(f"  Eval  examples: {len(eval_dataset)}")

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    tokenizer = load_tokenizer(args.model_name)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    print("Loading model ...")
    model = build_model_for_trl(
        args.model_name,
        use_bf16=True,
        use_4bit=args.use_4bit,
    )

    # ------------------------------------------------------------------
    # DPOConfig  (hyperparameters from proposal Section 2.7, Method 2)
    # ------------------------------------------------------------------
    from trl import DPOTrainer, DPOConfig
    import inspect

    # max_prompt_length was added to DPOConfig in TRL 0.9.x.
    # Older installs accept it only as a DPOTrainer kwarg — check first.
    _dpo_config_params = set(inspect.signature(DPOConfig).parameters)
    _dpo_config_kwargs = dict(
        output_dir=args.output_dir,
        # Reproducibility
        seed=args.seed,
        # Optimisation
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        # DPO-specific
        beta=args.beta,
        # Sequence lengths
        max_length=args.max_length,
        # Precision
        bf16=torch.cuda.is_bf16_supported(),
        fp16=(
            not torch.cuda.is_bf16_supported()
            and torch.cuda.is_available()
        ),
        # Memory: recompute activations on the backward pass instead of
        # storing them.  Cuts activation memory ~60% at ~20% extra compute.
        # Required for DPO on a single L4 (22 GB): each step runs both a
        # policy and a reference forward pass over chosen+rejected sequences.
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Logging — §1.6: save TensorBoard logs under ./runs/<run_name>/
        logging_steps=10,
        logging_dir=f"./runs/{args.output_dir}",
        save_steps=100,
        eval_strategy="epoch" if eval_dataset is not None else "no",
        report_to=["tensorboard"],
        remove_unused_columns=False,
    )
    if "max_prompt_length" in _dpo_config_params:
        _dpo_config_kwargs["max_prompt_length"] = args.max_prompt_length
    else:
        print(
            "  [INFO] This TRL version does not support max_prompt_length "
            "in DPOConfig; will pass it to DPOTrainer directly."
        )

    training_args = DPOConfig(**_dpo_config_kwargs)

    # ------------------------------------------------------------------
    # DPOTrainer
    #
    # ref_model=None: TRL uses the base weights of the PEFT model for the
    # reference forward pass by calling model.disable_adapter().  This
    # means we load the model once instead of twice.
    #
    # processing_class vs tokenizer: TRL renamed the parameter from
    # "tokenizer" to "processing_class" in 0.12.  We try the new name
    # first and fall back gracefully for older installs.
    # ------------------------------------------------------------------
    trainer_kwargs = dict(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=LORA_CONFIG,
    )
    # Pass max_prompt_length directly to DPOTrainer for older TRL versions
    # that don't support it in DPOConfig.
    if "max_prompt_length" not in _dpo_config_params:
        _trainer_params = set(inspect.signature(DPOTrainer.__init__).parameters)
        if "max_prompt_length" in _trainer_params:
            trainer_kwargs["max_prompt_length"] = args.max_prompt_length

    try:
        trainer = DPOTrainer(**trainer_kwargs, processing_class=tokenizer)
    except TypeError:
        # TRL < 0.12 uses "tokenizer" instead of "processing_class".
        trainer = DPOTrainer(**trainer_kwargs, tokenizer=tokenizer)

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    print("Starting DPO training ...")
    print(f"  beta        = {args.beta}  (KL penalty)")
    print(f"  lr          = {args.lr}")
    print(f"  epochs      = {args.epochs}")
    print(f"  max_length  = {args.max_length}  (prompt + code response)")
    print()
    print("  DPO is OFFLINE: no new code is generated during training.")
    print(
        "  The model learns from the fixed "
        "(prompt, chosen, rejected) triplets."
    )
    print()

    trainer.train()

    print(f"\nTraining complete. Saving to {args.output_dir} ...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
