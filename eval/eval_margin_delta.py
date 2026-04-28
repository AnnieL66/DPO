import argparse
import json
import os
import statistics
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def load_policy(model_path, dtype):
    tok_path = model_path
    adapter_cfg = os.path.join(model_path, "adapter_config.json") if os.path.isdir(model_path) else ""
    is_peft = os.path.exists(adapter_cfg)

    if is_peft:
        with open(adapter_cfg, "r", encoding="utf-8") as f:
            base_name = json.load(f)["base_model_name_or_path"]
        tok_path = base_name
        base = AutoModelForCausalLM.from_pretrained(
            base_name, torch_dtype=dtype, trust_remote_code=True
        ).cuda().eval()
        model = PeftModel.from_pretrained(base, model_path)
        model = model.merge_and_unload().eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, trust_remote_code=True
        ).cuda().eval()

    tok = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


@torch.no_grad()
def sequence_logprob(model, tokenizer, prompt, response):
    full = prompt + response
    full_ids = tokenizer(full, return_tensors="pt").input_ids.to(model.device)
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    prompt_len = prompt_ids.shape[1]

    logits = model(full_ids).logits
    log_probs = F.log_softmax(logits[:, :-1].float(), dim=-1)
    targets = full_ids[:, 1:]
    token_lp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    response_lp = token_lp[:, prompt_len - 1:]
    return response_lp.sum().item()


def margins(model, tok, pairs):
    out = []
    for i, p in enumerate(pairs):
        lp_c = sequence_logprob(model, tok, p["prompt"], p["chosen"])
        lp_r = sequence_logprob(model, tok, p["prompt"], p["rejected"])
        out.append(lp_c - lp_r)
        if (i + 1) % 50 == 0:
            print(f"{i+1}/{len(pairs)} pairs done")
    return out


def summarize(xs):
    return {
        "mean": sum(xs) / len(xs),
        "median": statistics.median(xs),
        "std": statistics.pstdev(xs),
        "min": min(xs),
        "max": max(xs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--trained_model", required=True)
    ap.add_argument("--eval_file", default="hh_eval_500.jsonl")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    pairs = [json.loads(line) for line in open(args.eval_file)]

    print(f"Loading base model: {args.base_model}")
    base_model, base_tok = load_policy(args.base_model, dtype)
    base_margins = margins(base_model, base_tok, pairs)
    del base_model
    torch.cuda.empty_cache()

    print(f"Loading trained model: {args.trained_model}")
    trained_model, trained_tok = load_policy(args.trained_model, dtype)
    trained_margins = margins(trained_model, trained_tok, pairs)

    deltas = [t - b for t, b in zip(trained_margins, base_margins)]
    result = {
        "base_model": args.base_model,
        "trained_model": args.trained_model,
        "eval_file": args.eval_file,
        "n_pairs": len(pairs),
        "base_preference_accuracy": sum(x > 0 for x in base_margins) / len(base_margins),
        "trained_preference_accuracy": sum(x > 0 for x in trained_margins) / len(trained_margins),
        "base_margin": summarize(base_margins),
        "trained_margin": summarize(trained_margins),
        "delta_trained_minus_base": summarize(deltas),
        "fraction_delta_positive": sum(x > 0 for x in deltas) / len(deltas),
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
