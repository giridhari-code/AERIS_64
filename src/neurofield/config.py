"""Configuration management for NeuroField v2."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class ModelConfig:
    d_model: int = 256
    d_k: int = 64
    d_v: int = 64
    n_skills: int = 8
    top_k: int = 2
    k_max: int = 4
    dendrite_window: int = 3
    vocab_size: int = 32000
    max_seq_len: int = 2048
    tie_embeddings: bool = True
    dropout: float = 0.0


@dataclass
class TrainingConfig:
    batch_size: int = 8
    grad_accum_steps: int = 1      # effective batch = batch_size * grad_accum_steps
    precision: str = "fp32"        # "fp32" | "bf16" (autocast; no GradScaler needed)
    seq_len: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_steps: int = 100_000
    grad_clip: float = 1.0
    lambda_err: float = 0.1
    lambda_bal: float = 0.01
    replay_every: int = 75
    replay_eta: float = 0.02
    truncate_write_window: int = 32
    device: str = "cuda"
    seed: int = 42
    log_every: int = 50
    eval_every: int = 500
    checkpoint_every: int = 2000
    output_dir: str = "runs/neurofield_v2"


@dataclass
class DataConfig:
    train_path: Optional[str] = None
    val_path: Optional[str] = None
    tokenizer: str = "byte"
    hf_tokenizer_name: Optional[str] = None


@dataclass
class SafetyConfig:
    enable_audit: bool = True
    max_write_norm: float = 10.0
    max_memory_norm: float = 50.0
    gate_cap: float = 1.0
    anomaly_z_threshold: float = 4.0
    freeze_on_anomaly: bool = False
    per_session_state: bool = True
    truncate_write_window: int = 32


@dataclass
class AblationConfig:
    """
    Runtime module switches for contribution measurement (Section 7 protocol).
    Default = production behaviour (all True). Training / serving leave this alone.
    """
    use_fast_memory: bool = True
    use_slow_memory: bool = True
    use_neuromodulator: bool = True
    use_metacognition: bool = True
    # When use_neuromodulator is False, gate is fixed to this scalar in (0, 1].
    fixed_gate: float = 0.5
    # When use_metacognition is False, field steps are fixed to this integer ≥ 1.
    fixed_k: int = 1


@dataclass
class LoggingConfig:
    level: str = "INFO"
    wandb: bool = False
    wandb_project: str = "neurofield-v2"


@dataclass
class NeuroFieldConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "NeuroFieldConfig":
        return cls(
            model=ModelConfig(**d.get("model", {})),
            training=TrainingConfig(**d.get("training", {})),
            data=DataConfig(**d.get("data", {})),
            safety=SafetyConfig(**d.get("safety", {})),
            ablation=AblationConfig(**d.get("ablation", {})),
            logging=LoggingConfig(**d.get("logging", {})),
        )


def load_config(path: str | Path | None = None) -> NeuroFieldConfig:
    """Load config from YAML. Falls back to defaults if path is None."""
    if path is None:
        return NeuroFieldConfig()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return NeuroFieldConfig.from_dict(raw)


def save_config(cfg: NeuroFieldConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_dict(), f, default_flow_style=False, sort_keys=False)


def model_preset(name: str = "tiny") -> ModelConfig:
    """Load a named size preset (tiny / small / medium) for capacity scaling."""
    import yaml
    path = Path(__file__).resolve().parents[2] / "configs" / "presets.yaml"
    if not path.is_file():
        # fallback relative to cwd
        path = Path("configs/presets.yaml")
    raw = {}
    if path.is_file():
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    block = raw.get(name) or raw.get("tiny") or {}
    fields = ModelConfig.__dataclass_fields__
    kwargs = {k: block[k] for k in fields if k in block}
    return ModelConfig(**kwargs)
