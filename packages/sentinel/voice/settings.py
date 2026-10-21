"""Production settings — all values from environment. No hardcoded secrets or paths."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional


def _env(key: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(key)
    if v is None or v == "":
        return default
    return v


def _env_int(key: str, default: int) -> int:
    v = _env(key)
    return int(v) if v is not None else default


def _env_float(key: str, default: float) -> float:
    v = _env(key)
    return float(v) if v is not None else default


def _env_bool(key: str, default: bool) -> bool:
    v = _env(key)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    """Runtime settings for Indo-Voice-3 services."""

    # Paths
    checkpoint_dir: str = field(default_factory=lambda: _env("INDO_CHECKPOINT", "checkpoints/AERIS_main") or "checkpoints/AERIS_main")
    audio_weights: Optional[str] = field(default_factory=lambda: _env("INDO_AUDIO_WEIGHTS"))
    device: str = field(default_factory=lambda: _env("INDO_DEVICE", "cpu") or "cpu")

    # STT / TTS
    stt_provider: str = field(default_factory=lambda: _env("INDO_STT_PROVIDER", "auto") or "auto")
    tts_provider: str = field(default_factory=lambda: _env("INDO_TTS_PROVIDER", "auto") or "auto")
    whisper_size: str = field(default_factory=lambda: _env("INDO_WHISPER_SIZE", "base") or "base")
    stt_device: str = field(default_factory=lambda: _env("INDO_STT_DEVICE", "cpu") or "cpu")
    tts_model: Optional[str] = field(default_factory=lambda: _env("INDO_TTS_MODEL"))

    # Generation
    max_new_tokens: int = field(default_factory=lambda: _env_int("INDO_MAX_NEW_TOKENS", 128))
    temperature: float = field(default_factory=lambda: _env_float("INDO_TEMPERATURE", 0.7))
    top_p: float = field(default_factory=lambda: _env_float("INDO_TOP_P", 0.9))

    # Language
    primary_language: str = field(default_factory=lambda: _env("INDO_PRIMARY_LANG", "hi") or "hi")
    enable_hinglish: bool = field(default_factory=lambda: _env_bool("INDO_HINGLISH", True))

    # Server
    host: str = field(default_factory=lambda: _env("INDO_HOST", "0.0.0.0") or "0.0.0.0")
    port: int = field(default_factory=lambda: _env_int("INDO_PORT", 8000))
    cors_origins: str = field(default_factory=lambda: _env("INDO_CORS_ORIGINS", "*") or "*")
    log_level: str = field(default_factory=lambda: _env("INDO_LOG_LEVEL", "INFO") or "INFO")

    # Audio I/O
    sample_rate: int = field(default_factory=lambda: _env_int("INDO_SAMPLE_RATE", 16000))
    max_audio_seconds: float = field(default_factory=lambda: _env_float("INDO_MAX_AUDIO_SEC", 30.0))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
