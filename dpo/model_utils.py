"""
model_utils.py
--------------
Load the policy model (with LoRA) and the frozen reference model from the
same Qwen2.5-Coder-1.5B-Instruct SFT checkpoint.

Why two separate model objects?
--------------------------------
DPO needs simultaneous access to:
  pi_theta  — the model being trained (policy)
  pi_ref    — the frozen SFT checkpoint (reference)

pi_ref must never change weights.  We load it as a separate object, freeze
every parameter, and never pass it an optimizer.  This is different from
LoRA-based tricks that toggle adapters: we want a clean, independent copy
so there is zero risk of accidentally updating the reference.

Memory note
-----------
Two copies of a 1.5B model is approximately 6 GB in bfloat16.  An L4 has
22 GB of VRAM.  However, TRL's DPOTrainer uses ref_model=None with a PEFT
model and calls model.disable_adapter() for the reference forward pass, so
only ONE copy of the base weights is loaded.  The dominant memory cost is
activations: DPO processes chosen and rejected sequences in the same step,
and runs both a policy and a reference forward pass, effectively 4× the
activation memory of standard SFT.  Keep per_device_train_batch_size ≤ 2
on a single L4 at max_length=768, and compensate with gradient_accumulation.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType


def disable_dropout(model: nn.Module) -> None:
    """
    Set dropout probability to 0 on every Dropout layer in the model.

    DPO computes log pi_theta(y|x) and log pi_ref(y|x) in separate forward
    passes.  If dropout is active, the same input produces different logits
    each call, adding stochastic noise to the log-ratios that is unrelated
    to the preference signal.  Disabling dropout makes every pass
    deterministic and ensures the log-ratios reflect only the model weights.

    This is applied to the policy so that chosen and rejected log-probs
    within the same concatenated forward pass are computed under identical
    stochastic conditions.  The reference model is in eval() mode anyway,
    which also disables dropout, but we call this on it for clarity.
    """
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0


# LoRA config from the project proposal (Section 2.3).
# lora_dropout is set to 0.0 because disable_dropout() is always called
# immediately after wrapping the model.  Declaring a non-zero value and
# then zeroing it is misleading, so we keep the config consistent with
# the actual runtime behaviour.
LORA_CONFIG = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.0,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    task_type=TaskType.CAUSAL_LM,
    bias="none",
)


def load_tokenizer(
    model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )
    # Qwen models use EOS as the pad token by default.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Right-padding keeps response tokens at the same positions for all
    # sequences in a batch, so the causal-LM shift in _batch_logps is
    # consistent across examples.
    tokenizer.padding_side = "right"
    return tokenizer


def load_policy_model(
    model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    use_bf16: bool = True,
    use_4bit: bool = False,
) -> torch.nn.Module:
    """
    Load the base model and wrap with LoRA adapters.

    Only the LoRA parameters are trainable (~0.3% of 1.5B = ~4.5M params).
    All base model weights are frozen by PEFT automatically.
    The model is returned in training mode so the optimizer can update the
    LoRA adapter parameters from the first step.
    """
    torch_dtype = torch.bfloat16 if use_bf16 else torch.float32

    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )

    policy_model = get_peft_model(base_model, LORA_CONFIG)
    policy_model.print_trainable_parameters()
    disable_dropout(policy_model)

    # Explicitly set to training mode.  from_pretrained returns the model
    # in train mode by default, but making this explicit avoids any
    # ambiguity when the function is called after eval() has been set
    # elsewhere in a script.
    policy_model.train()
    return policy_model


def load_ref_model(
    model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    use_bf16: bool = True,
) -> torch.nn.Module:
    """
    Load the reference model (pi_ref): same checkpoint, NO LoRA, ALL frozen.

    This is the SFT checkpoint that anchors the KL divergence in the DPO
    objective.  It never receives gradient updates.
    """
    torch_dtype = torch.bfloat16 if use_bf16 else torch.float32

    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    # Hard-freeze: no parameter will ever accumulate a gradient.
    for param in ref_model.parameters():
        param.requires_grad = False

    disable_dropout(ref_model)
    ref_model.eval()
    return ref_model


def load_models_and_tokenizer(
    model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    use_bf16: bool = True,
    use_4bit: bool = False,
):
    """Convenience loader that returns (tokenizer, policy_model, ref_model)."""
    tokenizer    = load_tokenizer(model_name)
    policy_model = load_policy_model(
        model_name, use_bf16=use_bf16, use_4bit=use_4bit
    )
    ref_model    = load_ref_model(model_name, use_bf16=use_bf16)
    return tokenizer, policy_model, ref_model
