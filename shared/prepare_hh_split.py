"""
Generate deterministic train/eval splits for hh-rlhf.
Run this script in both repos. Output files must match byte-for-byte
(verify with `md5sum hh_train_3k.jsonl hh_eval_500.jsonl` after running).

Run: python shared/prepare_hh_split.py
"""
import json
from datasets import load_dataset

SEED = 42
N_TRAIN = 3000
N_EVAL = 500
DATASET_REVISION = "09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa"


def parse_hh_example(text: str):
    """Split an hh-rlhf transcript into (prompt, final_response).

    The prompt is everything up to and including the last
    '\\n\\nAssistant:'. The final response is everything after it.
    """
    marker = "\n\nAssistant:"
    idx = text.rfind(marker)
    if idx == -1:
        return None
    prompt = text[: idx + len(marker)]
    response = text[idx + len(marker):].strip()
    return prompt, response


def to_pair(row):
    pc = parse_hh_example(row["chosen"])
    pr = parse_hh_example(row["rejected"])
    if pc is None or pr is None:
        return None
    prompt_c, chosen = pc
    prompt_r, rejected = pr
    # The prompt must be identical in the chosen/rejected transcripts.
    if prompt_c != prompt_r:
        return None
    if not chosen or not rejected:
        return None
    return {"prompt": prompt_c, "chosen": chosen, "rejected": rejected}


def build_split(split_name, n, out_path):
    ds = load_dataset("Anthropic/hh-rlhf", revision=DATASET_REVISION, split=split_name)
    ds = ds.shuffle(seed=SEED)
    pairs, i = [], 0
    while len(pairs) < n and i < len(ds):
        p = to_pair(ds[i])
        if p is not None:
            pairs.append(p)
        i += 1
    assert len(pairs) == n, (
        f"only got {len(pairs)} valid pairs, wanted {n}"
    )
    with open(out_path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"Wrote {len(pairs)} pairs to {out_path}")


if __name__ == "__main__":
    build_split("train", N_TRAIN, "hh_train_3k.jsonl")
    build_split("test", N_EVAL, "hh_eval_500.jsonl")
