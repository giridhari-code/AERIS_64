"""Shared pipeline context passed to every stage."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch

from sentinel.config import SENTINELConfig
from sentinel.model import SENTINEL

logger = logging.getLogger(__name__)


@dataclass
class PipelineContext:
    cfg: SENTINELConfig
    model: SENTINEL
    device: torch.device
    output_dir: Path
    stage_cfg: dict[str, Any] = field(default_factory=dict)
    # data
    train_bin: Optional[str] = None
    val_bin: Optional[str] = None
    sft_path: Optional[str] = None          # jsonl or txt
    prefs_path: Optional[str] = None        # preference jsonl
    prompts_path: Optional[str] = None      # RL prompts
    reasoning_path: Optional[str] = None
    # runtime
    ref_model: Optional[SENTINEL] = None    # frozen reference for DPO/PPO/GRPO
    reward_model: Optional[SENTINEL] = None
    lora_enabled: bool = False
    world_size: int = 1
    rank: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def stage_dir(self, name: str) -> Path:
        d = self.output_dir / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_model(self, name: str) -> Path:
        from sentinel.checkpoint.io import save_checkpoint

        out = self.stage_dir(name)
        save_checkpoint(
            out,
            self.model,
            config=self.cfg.model.__dict__.copy(),
            meta={"stage": name, "pipeline": True},
        )
        # also torch checkpoint for resume
        ckpt = out / "checkpoint.pt"
        torch.save(
            {
                "model": self.model.state_dict(),
                "config": self.cfg.to_dict(),
                "stage": name,
            },
            ckpt,
        )
        if self.is_main:
            logger.info("saved stage=%s → %s", name, out)
        return out

    def freeze_as_ref(self) -> None:
        """Clone current weights as frozen reference policy."""
        import copy

        self.ref_model = copy.deepcopy(self.model)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad_(False)
        if self.is_main:
            logger.info("reference policy frozen (%d params)", sum(p.numel() for p in self.ref_model.parameters()))
