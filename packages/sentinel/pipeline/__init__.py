"""Company-standard multi-stage training pipeline.

Stages (order is typical; any subset can be run):
  pretrain → midtrain → long_context → sft → reasoning
  → preference_bt (Bradley-Terry RM) → dpo / orpo / lora_pref
  → grpo → rlhf_ppo → opd (online preference distillation)

Entry:
  python -m sentinel.cli pipeline --config configs/pipeline/default.yaml
  python -m sentinel.cli pipeline --stages sft,dpo --data-dir data/tokens
"""

from __future__ import annotations

from sentinel.pipeline.registry import STAGES, list_stages
from sentinel.pipeline.runner import run_pipeline

__all__ = ["STAGES", "list_stages", "run_pipeline"]
