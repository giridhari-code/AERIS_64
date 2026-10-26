"""Configuration management for SENTINEL v2."""

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
    # True = only run selected top-k skill MLPs (faster; still AERIS top-k router)
    # False = run all skill MLPs every step (denser grads on every skill)
    sparse_skill_exec: bool = True
    k_max: int = 5
    dendrite_window: int = 3
    vocab_size: int = 32000
    max_seq_len: int = 3072
    context_slots: int = 0  # 0 => auto max_seq_len // truncate_window
    slow_compact: bool = True
    tie_embeddings: bool = True
    dropout: float = 0.0
    # --- Global Workspace (J-space) ---
    jspace_dim: int = 64
    jspace_slots: int = 16
    jspace_decay: float = 0.85
    jspace_write_threshold: float = 0.35
    jspace_sparsity: float = 0.25
    # --- Looped NeuralField ---
    field_reinject_input: bool = True
    field_loop_residual_scale: bool = True
    # --- Consciousness access layer ---
    consciousness_ignition_threshold: float = 0.55
    consciousness_access_threshold: float = 0.35
    consciousness_deliberate_threshold: float = 0.70
    # --- Inference / knowledge ---
    gk_cache_capacity: int = 128
    gk_cache_dim: int = 256
    gk_cache_min_score: float = 0.25
    heal_every_tokens: int = 64
    consciousness_gate_boost: float = 0.2
    # --- Hierarchical predictors (multi-timescale error) ---
    hierarchical_predict: bool = True
    pred_mid_horizon: int = 4
    pred_coarse_horizon: int = 16
    lambda_err_mid: float = 0.05
    lambda_err_coarse: float = 0.02


@dataclass
class TrainingConfig:
    batch_size: int = 8
    grad_accum_steps: int = 1      # effective batch = batch_size * grad_accum_steps
    precision: str = "bf16"        # "fp32" | "bf16" (autocast; no GradScaler needed)
    seq_len: int = 256
    lr: float = 5e-4               # safer default with memory + error loss
    weight_decay: float = 0.01
    warmup_steps: int = 200
    max_steps: int = 100_000
    grad_clip: float = 1.0
    lambda_err: float = 0.1
    lambda_err_warmup_steps: int = 2000  # ramp lambda_err 0 → value (avoids early embed collapse)
    lambda_bal: float = 0.01
    replay_every: int = 75
    replay_eta: float = 0.02
    truncate_write_window: int = 24  # smaller segment → more slow slots (yaad granularity)
    memory_lr_scale: float = 0.5   # fast/slow mem + neuromod get lr * scale
    ema_decay: float = 0.999       # 0 = disable EMA; eval uses EMA weights when > 0
    device: str = "cuda"
    seed: int = 42
    log_every: int = 50
    eval_every: int = 500
    checkpoint_every: int = 2000
    output_dir: str = "runs/sentinel_v2"


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
    max_memory_norm: float = 50.0  # fast/slow heal clamp (single source of truth)
    gate_cap: float = 1.0
    gate_floor: float = 0.05      # train-time minimum write gate (anti-collapse)
    gate_target: float = 0.35     # soft target for gate_mean regularizer
    lambda_gate: float = 0.01     # weight for (gate_mean - target)^2
    lambda_ent: float = 0.005     # weight for low router-entropy penalty
    slow_segment_len: int = 0     # 0 => auto max(4, min(16, truncate//4))
    anomaly_z_threshold: float = 4.0
    freeze_on_anomaly: bool = True  # paper §7: anomaly → g=0 (serve path)
    freeze_after_alarms: int = 3    # consecutive/cumulative z-alarms before freeze
    anomaly_min_history: int = 30   # samples before z-score is trusted
    per_session_state: bool = True
    truncate_write_window: int = 24


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
    wandb_project: str = "sentinel-v2"


@dataclass
class SENTINELConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SENTINELConfig":
        """Build config from nested dict; ignore unknown keys (forward-compatible YAML)."""

        def _filter(dc_cls: type, section: dict[str, Any] | None) -> dict[str, Any]:
            raw = section or {}
            if not isinstance(raw, dict):
                return {}
            fields = getattr(dc_cls, "__dataclass_fields__", {})
            return {k: v for k, v in raw.items() if k in fields}

        return cls(
            model=ModelConfig(**_filter(ModelConfig, d.get("model"))),
            training=TrainingConfig(**_filter(TrainingConfig, d.get("training"))),
            data=DataConfig(**_filter(DataConfig, d.get("data"))),
            safety=SafetyConfig(**_filter(SafetyConfig, d.get("safety"))),
            ablation=AblationConfig(**_filter(AblationConfig, d.get("ablation"))),
            logging=LoggingConfig(**_filter(LoggingConfig, d.get("logging"))),
        )


def load_config(path: str | Path | None = None) -> SENTINELConfig:
    """Load config from YAML. Falls back to defaults if path is None."""
    if path is None:
        return SENTINELConfig()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return SENTINELConfig.from_dict(raw)


def save_config(cfg: SENTINELConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_dict(), f, default_flow_style=False, sort_keys=False)


def model_preset(name: str = "aeris_109m") -> ModelConfig:
    """Load size preset (product default: aeris_109m)."""
    import yaml
    path = Path(__file__).resolve().parents[2] / "configs" / "presets.yaml"
    if not path.is_file():
        # fallback relative to cwd
        path = Path("configs/presets.yaml")
    raw = {}
    if path.is_file():
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    block = raw.get(name) or raw.get("aeris_109m") or (next(iter(raw.values())) if raw else {}) or {}
    fields = ModelConfig.__dataclass_fields__
    kwargs = {k: block[k] for k in fields if k in block}
    return ModelConfig(**kwargs)

# Backward-compatible alias
NeuroFieldConfig = SENTINELConfig

