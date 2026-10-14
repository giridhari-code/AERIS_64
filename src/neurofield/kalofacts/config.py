"""Configuration for Kalofacts (persistent interactive artifacts)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class KalofactsConfig:
    enabled: bool = True
    store_dir: str = "artifacts/kalofacts"
    max_content_bytes: int = 2_000_000
    max_versions: int = 50
    max_artifacts_per_session: int = 100
    # Auto-open when content length exceeds this (characters)
    auto_trigger_chars: int = 400
    allow_public_share: bool = False

    @classmethod
    def from_env(cls) -> "KalofactsConfig":
        return cls(
            enabled=_env_bool("NEUROFIELD_KALOFACTS", True),
            store_dir=os.environ.get("NEUROFIELD_KALOFACTS_DIR") or "artifacts/kalofacts",
            max_content_bytes=_env_int("NEUROFIELD_KALOFACTS_MAX_BYTES", 2_000_000),
            max_versions=_env_int("NEUROFIELD_KALOFACTS_MAX_VERSIONS", 50),
            max_artifacts_per_session=_env_int("NEUROFIELD_KALOFACTS_MAX_PER_SESSION", 100),
            auto_trigger_chars=_env_int("NEUROFIELD_KALOFACTS_AUTO_CHARS", 400),
            allow_public_share=_env_bool("NEUROFIELD_KALOFACTS_PUBLIC_SHARE", False),
        )


_CFG: Optional[KalofactsConfig] = None


def load_kalofacts_config(force: bool = False) -> KalofactsConfig:
    global _CFG
    if _CFG is None or force:
        _CFG = KalofactsConfig.from_env()
    return _CFG


def store_root() -> Path:
    cfg = load_kalofacts_config()
    root = Path(cfg.store_dir)
    root.mkdir(parents=True, exist_ok=True)
    return root
