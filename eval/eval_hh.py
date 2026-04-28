"""
eval/eval_hh.py
---------------
Evaluate a checkpoint on the hh-rlhf eval set (hh_eval_500.jsonl).

Computes metrics M1, M2, M3 from the project spec:
  M1 — Preference accuracy (sum log-prob)
  M2 — RM score of generations (requires --rm_path)
  M3 — Mean response length

Usage
-----
# Baseline (no RM scoring):
python eval/eval_hh.py \
    --model_path Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --eval_file hh_eval_500.jsonl \
    --output eval_results_base.json

# With RM scoring:
python eval/eval_hh.py \
    --model_path ./qwen-coder-dpo \
    --eval_file hh_eval_500.jsonl \
    --rm_path ./rm-checkpoint \
    --output eval_results_dpo.json
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def _load_model(model_path: str, use_4bit: bool = False):
    """
    Load a policy model from a HuggingFace model name or local path.

    If the path contains adapter_config.json (a LoRA adapter checkpoint saved
    by TRL's trainer.save_model()), the base model is loaded from the path
    recorded in adapter_config.json and the adapter is merged in-memory before
    returning.  This lets the eval scripts accept the raw training output
    directory (e.g. ./qwen-coder-dpo-hh) without a separate merge step.
    """
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    adapter_cfg = os.path.join(model_path, "adapter_config.json") if os.path.isdir(model_path) else ""
    is_peft = os.path.exists(adapter_cfg)

    if is_peft:
        with open(adapter_cfg) as f:
            base_name = json.load(f)["base_model_name_or_path"]
        print(f"  Detected LoRA adapter. Base model: {base_name}")
        from peft import PeftModel
        if use_4bit:
            from transformers import BitsAndBytesConfig
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch_dtype)
            base = AutoModelForCausalLM.from_pretrained(
                base_name, quantization_config=bnb, device_map="auto", trust_remote_code=True
            )
        else:
            base = AutoModelForCausalLM.from_pretrained(
                base_name, torch_dtype=torch_dtype, device_map="auto", trust_remote_code=True
            )
        model = PeftModel.from_pretrained(base, model_path)
        model = model.merge_and_unload()
        print("  Adapter merged into base model.")
    else:
        if use_4bit:
            from transformers import BitsAndBytesConfig
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch_dtype)
            model = AutoModelForCausalLM.from_pretrained(
                model_path, quantization_config=bnb, device_map="auto", trust_remote_code=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch_dtype, device_map="auto", trust_remote_code=True
            )
    return model


# ---------------------------------------------------------------------------
# M1: Preference accuracy (sum log-prob)
# ---------------------------------------------------------------------------

@torch.no_grad()
def sequence_logprob(model, tokenizer, prompt: str, response: str) -> float:
    """Sum of token log-probabilities for `response` conditioned on `prompt`."""
    full = prompt + response
    full_ids = tokenizer(full, return_tensors="pt").input_ids.to(model.device)
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    prompt_len = prompt_ids.shape[1]

    logits = model(full_ids).logits
    log_probs = F.log_softmax(logits[:, :-1].float(), dim=-1)
    targets = full_ids[:, 1:]
    token_lp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    # The causal shift means response tokens start at index prompt_len - 1.
    response_lp = token_lp[:, prompt_len - 1:]
    return response_lp.sum().item()


def preference_accuracy(model, tokenizer, pairs):
    correct = 0
    for i, p in enumerate(pairs):
        lp_c = sequence_logprob(model, tokenizer, p["prompt"], p["chosen"])
        lp_r = sequence_logprob(model, tokenizer, p["prompt"], p["rejected"])
        if lp_c > lp_r:
            correct += 1
        if (i + 1) % 50 == 0:
            print(f"  M1: {i+1}/{len(pairs)} pairs evaluated ...")
    return correct / len(pairs)


# ---------------------------------------------------------------------------
# M3: Mean response length
# ---------------------------------------------------------------------------

@torch.no_grad()
def mean_response_length(policy, policy_tok, prompts):
    lengths = []
    for prompt in prompts:
        ids = policy_tok(prompt, return_tensors="pt").to(policy.device)
        gen = policy.generate(
            **ids,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=policy_tok.eos_token_id,
        )
        n_new = gen.shape[1] - ids.input_ids.shape[1]
        lengths.append(n_new)
    return sum(lengths) / len(lengths)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   help="HF model name or local path to the policy checkpoint")
    p.add_argument("--eval_file", default="hh_eval_500.jsonl",
                   help="Path to hh_eval_500.jsonl")
    p.add_argument("--rm_path", default=None,
                   help="Path to the reward model (optional; skips M2 if absent)")
    p.add_argument("--out", default="eval_results.json",
                   help="Path to write the JSON results file")
    p.add_argument("--n_pairs", type=int, default=500,
                   help="Number of eval pairs to use (default: all 500)")
    p.add_argument("--use_4bit", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    # Load eval pairs
    with open(args.eval_file) as f:
        pairs = [json.loads(line) for line in f]
    pairs = pairs[: args.n_pairs]
    print(f"Loaded {len(pairs)} eval pairs from {args.eval_file}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print(f"Loading policy: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = _load_model(args.model, use_4bit=args.use_4bit)
    model.eval()

    results = {}

    print("Computing M1: preference accuracy ...")
    acc = preference_accuracy(model, tokenizer, pairs)
    results["M1_preference_accuracy"] = acc
    print(f"  M1 = {acc:.4f}")

    if args.rm_path is None:
        print("Computing M3: mean response length (no RM path given, skipping M2) ...")
        avg_len = mean_response_length(model, tokenizer, [p["prompt"] for p in pairs])
        results["M3_mean_response_length"] = avg_len
        print(f"  M3 = {avg_len:.1f} tokens")
    else:
        print(f"Loading reward model: {args.rm_path}")
        rm_tok = AutoTokenizer.from_pretrained(args.rm_path, trust_remote_code=True)
        rm = AutoModelForCausalLM.from_pretrained(
            args.rm_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        rm.eval()

        print("Computing M2 (RM scores) and M3 (response lengths) ...")
        prompts = [p["prompt"] for p in pairs]
        ids_list = [tokenizer(pr, return_tensors="pt").to(device) for pr in prompts]
        scores = []
        lengths = []
        for i, (prompt, ids) in enumerate(zip(prompts, ids_list)):
            gen = model.generate(
                **ids,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            n_new = gen.shape[1] - ids.input_ids.shape[1]
            lengths.append(n_new)
            text = tokenizer.decode(gen[0, ids.input_ids.shape[1]:], skip_special_tokens=True)
            rm_in = rm_tok(
                prompt + text, return_tensors="pt", truncation=True, max_length=1024
            ).to(device)
            scores.append(rm(**rm_in).logits.squeeze().item())
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(prompts)} done ...")

        sorted_scores = sorted(scores)
        results["M2_rm_score_mean"] = sum(scores) / len(scores)
        results["M2_rm_score_median"] = sorted_scores[len(sorted_scores) // 2]
        results["M3_mean_response_length"] = sum(lengths) / len(lengths)
        print(f"  M2 mean  = {results['M2_rm_score_mean']:.4f}")
        print(f"  M2 median= {results['M2_rm_score_median']:.4f}")
        print(f"  M3       = {results['M3_mean_response_length']:.1f} tokens")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {args.out}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
