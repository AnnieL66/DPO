"""
DPO pipeline for code-generation RLHF.

Modules
-------
dpo_loss    — core math: compute_log_probs, dpo_loss, training_step
model_utils — load policy (LoRA) and frozen reference models
data_utils  — dataset formatting and tokenisation helpers
train_trl   — TRL DPOTrainer entry-point
toy_example — toy batch that verifies loss direction
"""
