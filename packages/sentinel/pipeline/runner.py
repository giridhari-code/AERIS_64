"""Pipeline runner — executes ordered stages with shared context."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
import yaml

from sentinel.config import SENTINELConfig, load_config
from sentinel.model import SENTINEL
from sentinel.pipeline.context import PipelineContext
from sentinel.pipeline.registry import STAGES, list_stages

# side-effect: register all stages
import sentinel.pipeline.stages  # noqa: F401

logger = logging.getLogger(__name__)

DEFAULT_ORDER = [
    "pretrain",
    "midtrain",
    "long_context",
    "sft",
    "reasoning",
    "preference_bt",
    "dpo",
    "orpo",
    "lora_pref",
    "grpo",
    "rlhf_ppo",
    "opd",
]



def _load_weights_into_model(model: torch.nn.Module, path: str | Path) -> None:
    """Load weights from checkpoint_*.pt, model.pt, model.safetensors, or an export folder."""
    path = Path(path)
    if path.is_dir():
        candidates = [
            path / "model.safetensors",
            path / "checkpoint.pt",
            path / "checkpoint_final.pt",
            path / "checkpoint_latest.pt",
            path / "model.pt",
        ]
        candidates.extend(sorted(path.glob("checkpoint*.pt")))
        found = next((c for c in candidates if c.is_file()), None)
        if found is None:
            raise FileNotFoundError(
                f"No model.safetensors or checkpoint*.pt under directory: {path}"
            )
        path = found

    if not path.is_file():
        raise FileNotFoundError(f"resume path not found: {path}")

    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as e:
            raise ImportError("pip install safetensors") from e
        state = load_file(str(path))
        model.load_state_dict(state, strict=False)
        return

    if path.suffix in {".pt", ".pth"}:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            if any(isinstance(v, torch.Tensor) for v in ckpt.values()):
                # flat or model-keyed state dict
                state = ckpt.get("model", ckpt)
                if isinstance(state, dict) and "state_dict" in state and not any(
                    isinstance(v, torch.Tensor) for v in state.values()
                ):
                    state = state["state_dict"]
            elif "model" in ckpt:
                state = ckpt["model"]
            elif "state_dict" in ckpt:
                state = ckpt["state_dict"]
            else:
                state = ckpt
        else:
            raise ValueError(f"Unrecognized checkpoint format: {path}")
        model.load_state_dict(state, strict=False)
        return

    raise ValueError(
        f"Unsupported resume file type: {path} (use .pt, .safetensors, or export dir)"
    )


def run_pipeline(
    stages: Sequence[str] | None = None,
    config_path: str | Path | None = None,
    pipeline_yaml: str | Path | None = None,
    output_dir: str | Path = "runs/pipeline",
    train_bin: str | None = None,
    val_bin: str | None = None,
    sft_path: str | None = None,
    prefs_path: str | None = None,
    reasoning_path: str | None = None,
    device: str | None = None,
    resume_checkpoint: str | None = None,
    tokenizer_json: str | None = None,
) -> PipelineContext:
    """Run selected stages in order. Unknown stage names raise."""
    cfg = load_config(config_path)
    stage_cfg: dict[str, Any] = {}
    # CLI --stages wins over pipeline YAML when provided
    cli_stages = list(stages) if stages else None
    order = list(cli_stages) if cli_stages else list(DEFAULT_ORDER)

    if pipeline_yaml:
        raw = yaml.safe_load(Path(pipeline_yaml).read_text(encoding="utf-8")) or {}
        if cli_stages is None and raw.get("stages"):
            order = list(raw["stages"])
        elif cli_stages is not None and raw.get("stages"):
            logger.info(
                "CLI --stages overrides pipeline YAML stages (%s → %s)",
                raw.get("stages"),
                cli_stages,
            )
        stage_cfg = raw.get("stage_cfg") or raw.get("stage") or {}
        if raw.get("model") or raw.get("training"):
            from sentinel.config import SENTINELConfig as SC

            overlay = SC.from_dict(raw)
            if raw.get("model"):
                cfg.model = overlay.model
            if raw.get("training"):
                cfg.training = overlay.training

    for s in order:
        if s not in STAGES:
            raise ValueError(f"Unknown stage {s!r}. Known: {list_stages()}")

    dev = torch.device(
        device
        or cfg.training.device
        or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if dev.type == "cuda" and not torch.cuda.is_available():
        dev = torch.device("cpu")

    model = SENTINEL(cfg.model, cfg.safety)
    if resume_checkpoint:
        _load_weights_into_model(model, resume_checkpoint)
        logger.info("resumed weights from %s", resume_checkpoint)
    model.to(dev)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ctx = PipelineContext(
        cfg=cfg,
        model=model,
        device=dev,
        output_dir=out,
        stage_cfg=stage_cfg,
        train_bin=train_bin or cfg.data.train_path,
        val_bin=val_bin or cfg.data.val_path,
        sft_path=sft_path,
        prefs_path=prefs_path,
        reasoning_path=reasoning_path,
        tokenizer_json=tokenizer_json,
    )

    logger.info("pipeline stages: %s", " → ".join(order))
    for name in order:
        logger.info("---------- running %s ----------", name)
        STAGES[name](ctx)
    logger.info("pipeline complete → %s", out)
    return ctx
