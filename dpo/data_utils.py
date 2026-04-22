"""
data_utils.py
-------------
Dataset utilities for the DPO pipeline.

Supported datasets
------------------
  "hh"  — Anthropic Helpful-Harmless  (Anthropic/hh-rlhf)
  "shp" — Stanford Human Preferences  (stanfordnlp/SHP)
  custom — pass raw_data directly (list of plain-string triplets)

Data format (all datasets normalised to this)
---------------------------------------------
    {
        "prompt":   "<full conversation up to last Assistant:>",
        "chosen":   "<preferred response text only>",
        "rejected": "<non-preferred response text only>"
    }

Truncation modes
----------------
  keep_end   — HH: multi-turn conversations; preserve recent context
               by dropping from the LEFT of the prompt when over-length.
  keep_start — SHP: single question at the start; drop from the RIGHT.
"""

from typing import Dict, List, Optional

from datasets import Dataset as HFDataset

DATASET_TRUNCATION_MODE: Dict[str, str] = {
    "hh":       "keep_end",
    "hh_local": "keep_end",   # same format as hh; preserve recent context
    "shp":      "keep_start",
}

SUPPORTED_DATASETS: tuple = ("hh", "hh_local", "shp")

# Pin the same HH revision used by shared/prepare_hh_split.py so that
# training and eval data come from the identical snapshot of the dataset.
HH_DATASET_REVISION = "09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa"


# ---------------------------------------------------------------------------
# Sample data (plain string format, used for smoke-tests)
# ---------------------------------------------------------------------------

SAMPLE_PREFERENCE_DATA: List[Dict] = [
    {
        "prompt": "\n\nHuman: Write a Python function `second_largest(nums)`"
                  " that returns the second largest unique element in a list"
                  " of integers. Return None if fewer than 2 unique elements"
                  " exist.\n\nAssistant:",
        "chosen": (
            " def second_largest(nums):\n"
            "    unique = sorted(set(nums), reverse=True)\n"
            "    if len(unique) >= 2:\n"
            "        return unique[1]\n"
            "    return None"
        ),
        "rejected": (
            " def second_largest(nums):\n"
            "    nums.sort()\n"
            "    return nums[-2]"
        ),
    },
    {
        "prompt": "\n\nHuman: Write a Python function `is_palindrome(s)`"
                  " that returns True if the string s is a palindrome"
                  " (ignoring case and spaces).\n\nAssistant:",
        "chosen": (
            " def is_palindrome(s):\n"
            "    cleaned = s.replace(' ', '').lower()\n"
            "    return cleaned == cleaned[::-1]"
        ),
        "rejected": (
            " def is_palindrome(s):\n"
            "    for i in range(len(s)):\n"
            "        if s[i] != s[len(s)-1-i]:\n"
            "            return False\n"
            "    return True"
        ),
    },
    {
        "prompt": "\n\nHuman: Write a Python function `flatten(lst)`"
                  " that recursively flattens a nested list of integers"
                  " into a single list.\n\nAssistant:",
        "chosen": (
            " def flatten(lst):\n"
            "    result = []\n"
            "    for item in lst:\n"
            "        if isinstance(item, list):\n"
            "            result.extend(flatten(item))\n"
            "        else:\n"
            "            result.append(item)\n"
            "    return result"
        ),
        "rejected": (
            " def flatten(lst):\n"
            "    return [x for x in lst"
            " if not isinstance(x, list)]"
        ),
    },
]


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

def _extract_anthropic_prompt(prompt_and_response: str) -> str:
    """
    Return everything up to and including the last '\\n\\nAssistant:'.

    Uses rfind so multi-turn conversations are handled correctly:
    the *last* Assistant marker is where the scored response begins.
    Raises AssertionError if no marker is found so callers can skip
    malformed rows rather than silently slicing at position 0.
    """
    marker = "\n\nAssistant:"
    idx = prompt_and_response.rfind(marker)
    if idx == -1:
        raise AssertionError(
            "No '\\n\\nAssistant:' marker found. "
            "Is this an Anthropic HH row?"
        )
    return prompt_and_response[: idx + len(marker)]


def get_hh(split: str, silent: bool = False) -> List[Dict]:
    """
    Load Anthropic Helpful-Harmless (Anthropic/hh-rlhf).

    Returns a flat list of plain-string triplets:
        prompt   — full conversation up to the last '\\n\\nAssistant:'
        chosen   — preferred response text only
        rejected — dispreferred response text only

    The prompt is extracted from the ``chosen`` field. We then verify that
    the ``rejected`` field carries the identical prompt prefix before slicing
    it at the same boundary. In the standard HH dataset chosen and rejected
    always share the same conversation history, but checking defensively
    prevents silent misalignment if a row has any encoding or whitespace
    difference between the two fields.
    """
    from datasets import load_dataset
    import tqdm

    print(f"Loading Anthropic HH ({split} split) ...")
    dataset = load_dataset(
        "Anthropic/hh-rlhf", revision=HH_DATASET_REVISION, split=split
    )
    print(f"  {len(dataset)} rows loaded.")

    data: List[Dict] = []
    skipped = 0

    for row in tqdm.tqdm(dataset, desc="Processing HH", disable=silent):
        try:
            prompt = _extract_anthropic_prompt(row["chosen"])
        except AssertionError:
            skipped += 1
            continue

        # Verify that rejected carries the same prompt as chosen.
        # Without this check, slicing rejected at len(prompt) would silently
        # produce a mismatched (prompt, chosen, rejected) triplet whenever
        # the two fields diverge before the final Assistant turn.
        try:
            rejected_prompt = _extract_anthropic_prompt(row["rejected"])
        except AssertionError:
            skipped += 1
            continue

        if rejected_prompt != prompt:
            skipped += 1
            continue

        chosen   = row["chosen"][len(prompt):]
        rejected = row["rejected"][len(prompt):]
        data.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

    if skipped:
        print(
            f"  Warning: {skipped} rows skipped "
            f"(missing marker or prompt mismatch between chosen/rejected)."
        )
    print(f"  {len(data)} usable pairs.")
    return data


def get_hh_local(
    split: str,
    filepath: str,
    silent: bool = False,
) -> List[Dict]:
    """
    Load hh-rlhf preference pairs from a local JSONL file produced by
    shared/prepare_hh_split.py.

    The `split` argument is accepted for interface consistency with the other
    loaders but is ignored — the caller selects the file via `filepath`.

    Each line must be a JSON object with keys:
        prompt   — full conversation up to the last '\\n\\nAssistant:'
        chosen   — preferred response text only
        rejected — dispreferred response text only
    """
    import json

    if not filepath:
        raise ValueError(
            "get_hh_local requires a 'filepath' kwarg. "
            "Pass it via --train_file or --eval_file."
        )
    if not silent:
        print(f"Loading hh_local from {filepath} ...")
    data: List[Dict] = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    if not silent:
        print(f"  {len(data)} pairs loaded.")
    return data


def get_shp(
    split: str,
    min_score_ratio: float = 2.0,
    silent: bool = False,
) -> List[Dict]:
    """
    Load Stanford Human Preferences (stanfordnlp/SHP).

    Filters pairs where the preference signal is too weak
    (max_score / min_score < min_score_ratio).

    Both scores must be strictly positive before the ratio is computed.
    Reddit upvote scores can be zero or negative; dividing by a negative
    number produces a ratio with the wrong sign (e.g. scores +5 and -3
    give ratio -1.67, which is less than 2.0 and would be discarded even
    though the preference is clearly strong). Requiring score > 0 for both
    responses keeps the ratio semantically meaningful.
    """
    from datasets import load_dataset
    import tqdm

    print(f"Loading Stanford SHP ({split} split) ...")
    dataset = load_dataset("stanfordnlp/SHP", split=split)
    print(f"  {len(dataset)} rows before filtering.")

    data: List[Dict] = []
    skipped = 0

    for row in tqdm.tqdm(dataset, desc="Processing SHP", disable=silent):
        score_a, score_b = row["score_A"], row["score_B"]

        # Both scores must be strictly positive so the ratio is valid.
        # Zero causes ZeroDivisionError; negative causes a sign inversion
        # that makes the ratio filter reject strong preference pairs.
        if score_a <= 0 or score_b <= 0:
            skipped += 1
            continue

        if max(score_a, score_b) / min(score_a, score_b) < min_score_ratio:
            skipped += 1
            continue

        prompt = "\n\nHuman: " + row["history"] + "\n\nAssistant:"
        if row["labels"] == 1:
            chosen   = " " + row["human_ref_A"]
            rejected = " " + row["human_ref_B"]
        else:
            chosen   = " " + row["human_ref_B"]
            rejected = " " + row["human_ref_A"]

        data.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

    print(
        f"  {len(data)} pairs kept, "
        f"{skipped} skipped (non-positive scores or ratio < {min_score_ratio})."
    )
    return data


def load_dataset_by_name(
    name: str,
    split: str,
    **kwargs,
) -> List[Dict]:
    """Dispatch to the correct loader by dataset name."""
    if name == "hh":
        return get_hh(split, **kwargs)
    elif name == "hh_local":
        return get_hh_local(split, **kwargs)
    elif name == "shp":
        return get_shp(split, **kwargs)
    else:
        raise ValueError(
            f"Unknown dataset '{name}'. "
            f"Supported: {SUPPORTED_DATASETS}"
        )


# ---------------------------------------------------------------------------
# TRL-compatible dataset builder
# ---------------------------------------------------------------------------

def build_trl_dataset(
    raw_data: Optional[List[Dict]] = None,
    dataset_name: Optional[str] = None,
    hf_dataset_name: Optional[str] = None,
    split: str = "train",
    **loader_kwargs,
) -> HFDataset:
    """
    Return a HuggingFace Dataset ready for TRL's DPOTrainer.

    Priority order:
      1. dataset_name in ("hh", "shp") — use built-in loader with correct
         prompt extraction logic
      2. hf_dataset_name               — load directly from HF Hub
      3. raw_data                       — use the provided list
      4. (none)                         — fall back to SAMPLE_PREFERENCE_DATA

    TRL's DPOTrainer requires exactly three plain-string columns:
        "prompt", "chosen", "rejected"

    When loading from HF Hub (option 2), the presence of these columns is
    validated before returning. For example, the raw Anthropic/hh-rlhf
    dataset does NOT have a "prompt" column — it stores the full conversation
    inside "chosen" and "rejected". Passing it directly to DPOTrainer would
    produce a silent training error. Use dataset_name="hh" instead, which
    routes through get_hh() and performs the correct prompt extraction.
    """
    if dataset_name is not None:
        data = load_dataset_by_name(dataset_name, split, **loader_kwargs)
        return HFDataset.from_list(data)

    if hf_dataset_name is not None:
        from datasets import load_dataset
        ds = load_dataset(hf_dataset_name, split=split)

        # Validate that the dataset already has the three required columns.
        required = {"prompt", "chosen", "rejected"}
        missing = required - set(ds.column_names)
        if missing:
            raise ValueError(
                f"Dataset '{hf_dataset_name}' (split='{split}') is missing "
                f"required columns: {sorted(missing)}.\n"
                f"Available columns: {list(ds.column_names)}.\n"
                f"TRL's DPOTrainer expects plain-string columns "
                f"'prompt', 'chosen', and 'rejected'.\n"
                f"For Anthropic/hh-rlhf, pass dataset_name='hh' so that "
                f"the built-in loader extracts the prompt correctly."
            )
        return ds

    data = raw_data if raw_data is not None else SAMPLE_PREFERENCE_DATA

    for i, row in enumerate(data):
        for key in ("prompt", "chosen", "rejected"):
            if key not in row:
                raise ValueError(f"Row {i} missing key '{key}'")
            if not isinstance(row[key], str):
                raise TypeError(
                    f"Row {i}: '{key}' must be a str, got {type(row[key])}"
                )

    return HFDataset.from_list(data)
