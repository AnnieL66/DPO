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
import re
import subprocess
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _build_prompt(tokenizer, task_prompt: str) -> str:
    """
    Wrap the raw HumanEval task prompt for the model.

    Instruct models (e.g. Qwen2.5-Coder-*-Instruct) require input formatted
    via the chat template.  Without it they generate conversational text
    instead of code, collapsing pass@1 from ~65% to ~18%.

    Base models (no chat_template) receive the raw prompt directly, which is
    the standard HumanEval completion-style setup.
    """
    if getattr(tokenizer, "chat_template", None):
        messages = [{
            "role": "user",
            "content": (
                "Complete the body of the following Python function. "
                "Output only the code, no explanation:\n\n" + task_prompt
            ),
        }]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return task_prompt


def _extract_code(response: str) -> str:
    """
    Strip markdown fences and return plain Python code.

    evalplus.sanitize searches for 'def <entry_point>' inside the completion
    and returns "" when it cannot find it.  Instruct models return the function
    *body* (indented, no 'def' header), so sanitize always strips them to "".
    Instead we just remove the ```python ... ``` wrapper ourselves and pass the
    raw code straight to evalplus.evaluate.
    """
    blocks = re.findall(r"```(?:python)?\n?(.*?)```", response, re.DOTALL)
    if blocks:
        return blocks[-1]
    return response


def _stop_token_ids(tokenizer) -> list:
    """
    Collect EOS / end-of-turn token IDs for the model.

    Qwen models use <|im_end|> as the end-of-turn token.  Including it here
    stops generation at the natural end of the assistant turn instead of
    continuing into padding or hallucinated turns.
    """
    ids = [tokenizer.eos_token_id]
    for name in ["<|im_end|>", "<|endoftext|>", "<|EOT|>"]:
        tid = tokenizer.convert_tokens_to_ids(name)
        if tid and tid != tokenizer.unk_token_id and tid not in ids:
            ids.append(tid)
    return ids


@torch.no_grad()
def generate_completion(model, tokenizer, task_prompt: str, max_new_tokens: int = 512) -> str:
    """Greedy decode a single completion for one HumanEval task prompt."""
    prompt_text = _build_prompt(tokenizer, task_prompt)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=_stop_token_ids(tokenizer),
    )
    return tokenizer.decode(
        outputs[0, inputs.input_ids.shape[1]:], skip_special_tokens=True
    )


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
        completion = generate_completion(
            model, tokenizer, task["prompt"], args.max_new_tokens
        )
        samples.append({"task_id": task_id, "completion": _extract_code(completion)})
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(problems)} done ...")

    # Write raw samples.jsonl
    with open(samples_path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(f"Wrote {len(samples)} completions to {samples_path}")

    # ------------------------------------------------------------------
    # NOTE: We intentionally skip evalplus.sanitize here.
    # evalplus.sanitize searches for 'def <entry_point>' in the completion
    # and strips it to "" when not found.  Instruct models output the function
    # body (indented, no 'def' header), so sanitize always produces empty
    # completions → pass@1 = 0.0.  We run _extract_code() during generation
    # above to strip markdown fences; no further sanitization is needed.
    eval_samples_path = samples_path

    # ------------------------------------------------------------------
    # Evaluate pass@1
    # ------------------------------------------------------------------
    print("Running pass@1 evaluation ...")

    if use_evalplus:
        cmd = [
            sys.executable, "-m", "evalplus.evaluate",
            "--dataset", "humaneval",
            "--samples", eval_samples_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print("evalplus stderr:", result.stderr)
        # evalplus writes <samples_stem>_eval_results.json next to the samples file
        stem = os.path.splitext(eval_samples_path)[0]
        evalplus_results = stem + "_eval_results.json"
        pass_at_1 = None
        if os.path.exists(evalplus_results):
            with open(evalplus_results) as f:
                eval_data = json.load(f)
            he = eval_data.get("humaneval", eval_data)
            pass_at_1 = he.get("pass@1", None)
    else:
        from human_eval.evaluation import evaluate_functional_correctness
        eval_data = evaluate_functional_correctness(eval_samples_path)
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
