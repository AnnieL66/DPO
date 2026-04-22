"""
eval/eval_humaneval.py
----------------------
Generate greedy completions for all 164 HumanEval problems and report pass@1.

Uses the `evalplus` library (pip install evalplus) which extends the original
HumanEval benchmark with additional tests.  Falls back to the `human_eval`
library if evalplus is not installed.

Usage
-----
# Baseline:
python eval/eval_humaneval.py \
    --model_path Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --out results/baseline_humaneval.json

# DPO checkpoint:
python eval/eval_humaneval.py \
    --model ./qwen-coder-dpo \
    --out results/dpo_humaneval.json
"""

import argparse
import json
import os
import subprocess
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def make_prompt(task: dict) -> str:
    """
    Build the prompt for a HumanEval task.

    The task dict has a "prompt" field which is the function signature +
    docstring.  We pass it as-is; the model should complete the function body.
    """
    return task["prompt"]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_completion(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    """Greedy decode a single completion for one HumanEval prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    # Decode only the newly generated tokens (strip the prompt).
    completion = tokenizer.decode(
        outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True
    )
    return completion


def stop_at_function_boundary(completion: str) -> str:
    """
    Truncate the completion at the first top-level definition or class after
    the initial function body.  HumanEval tasks define one function; any
    subsequent `def` or `class` at the top level is outside the target scope.
    """
    lines = completion.split("\n")
    result = []
    for line in lines:
        # A new top-level def/class signals the end of the target function.
        if result and line and not line[0].isspace() and (
            line.startswith("def ") or line.startswith("class ")
        ):
            break
        result.append(line)
    return "\n".join(result)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   help="HF model name or local path to the policy checkpoint")
    p.add_argument("--out", default="results/humaneval_results.json",
                   help="Path to write the JSON results file")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--use_4bit", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    # Derive a samples directory from the output file location.
    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)
    samples_path = os.path.join(out_dir, "samples.jsonl")

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.use_4bit:
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=bnb, device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
    model.eval()

    # ------------------------------------------------------------------
    # Load HumanEval problems
    # ------------------------------------------------------------------
    try:
        from evalplus.data import get_human_eval_plus
        problems = get_human_eval_plus()
        print(f"Loaded {len(problems)} HumanEval+ problems via evalplus")
        use_evalplus = True
    except ImportError:
        try:
            from human_eval.data import read_problems
            problems = read_problems()
            print(f"Loaded {len(problems)} HumanEval problems via human-eval")
            use_evalplus = False
        except ImportError:
            print("ERROR: neither evalplus nor human-eval is installed.")
            print("Install with: pip install evalplus  OR  pip install human-eval")
            sys.exit(1)

    # ------------------------------------------------------------------
    # Generate completions
    # ------------------------------------------------------------------
    print(f"Generating completions for {len(problems)} problems ...")
    samples = []
    for i, (task_id, task) in enumerate(problems.items()):
        prompt = make_prompt(task)
        completion = generate_completion(model, tokenizer, prompt, args.max_new_tokens)
        completion = stop_at_function_boundary(completion)
        samples.append({"task_id": task_id, "completion": completion})
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(problems)} done ...")

    # Write samples.jsonl
    with open(samples_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(f"Wrote {len(samples)} completions to {samples_path}")

    # ------------------------------------------------------------------
    # Evaluate pass@1
    # ------------------------------------------------------------------
    print("Running pass@1 evaluation ...")
    results_path = args.out

    if use_evalplus:
        # evalplus CLI: evalplus.evaluate --dataset humaneval --samples <path>
        cmd = [
            sys.executable, "-m", "evalplus.evaluate",
            "--dataset", "humaneval",
            "--samples", samples_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print("evalplus stderr:", result.stderr)
        # evalplus writes results as <samples_stem>_eval_results.json
        # next to samples.jsonl (e.g. samples_eval_results.json).
        samples_stem = os.path.splitext(samples_path)[0]
        evalplus_results = samples_stem + "_eval_results.json"
        pass_at_1 = None
        if os.path.exists(evalplus_results):
            with open(evalplus_results) as f:
                eval_data = json.load(f)
            # evalplus reports pass@1 under the "humaneval" key
            he = eval_data.get("humaneval", eval_data)
            pass_at_1 = he.get("pass@1", None)
    else:
        from human_eval.evaluation import evaluate_functional_correctness
        eval_data = evaluate_functional_correctness(samples_path)
        pass_at_1 = eval_data.get("pass@1", None)

    summary = {
        "model": args.model,
        "n_problems": len(problems),
        "M4_humaneval_pass_at_1": pass_at_1,
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults written to {args.out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
