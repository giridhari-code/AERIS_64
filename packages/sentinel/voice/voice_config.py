"""Configuration for Indo-Voice-3 (SENTINEL + Voice extensions).

All values are environment-driven or passed explicitly.
No hardcoded secrets or magic numbers in runtime code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """Core SENTINEL model dimensions (kept compatible with original)."""

    vocab_size: int = 32000
    d_model: int = 768
    d_k: int = 64
    d_v: int = 64
    n_skills: int = 8
    top_k: int = 2
    k_max: int = 4
    dendrite_window: int = 8
    max_seq_len: int = 2048
    dropout: float = 0.1
    tie_embeddings: bool = True
    context_slots: int = 0          # 0 = auto-compute
    slow_compact: bool = True


@dataclass
class SafetyConfig:
    """Safety / training stability knobs."""

    truncate_write_window: int = 32
    slow_segment_len: int = 0       # 0 = auto
    gate_cap: float = 0.95
    gate_floor: float = 0.05
    gate_target: float = 0.35
    lambda_gate: float = 0.01
    lambda_ent: float = 0.005
    max_memory_norm: float = 10.0


@dataclass
class AblationConfig:
    """Ablation flags for research / debugging."""

    use_metacognition: bool = True
    use_neuromodulator: bool = True
    use_fast_memory: bool = True
    use_slow_memory: bool = True
    fixed_k: int = 2
    fixed_gate: float = 0.5


@dataclass
class AudioConfig:
    """Custom trainable audio encoder settings."""

    sample_rate: int = 16000
    n_mels: int = 80
    n_fft: int = 400
    hop_length: int = 160           # 10 ms at 16 kHz
    win_length: int = 400
    encoder_layers: int = 6
    encoder_dim: int = 512
    encoder_heads: int = 8
    encoder_ff_dim: int = 2048
    conv_channels: int = 256
    dropout: float = 0.1
    max_audio_seconds: float = 30.0


@dataclass
class VoiceConfig:
    """Indo-Voice-3 runtime configuration."""

    # Streaming
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    streaming_chunk_size: int = 4   # tokens per yield

    # Hindi / Hinglish
    primary_language: str = "hi"    # BCP-47
    enable_hinglish: bool = True
    language_hint_weight: float = 0.3

    # Session
    max_session_tokens: int = 4096
    consolidate_every_turns: int = 1

    # External TTS (optional fallback)
    tts_provider: str = "none"      # "none" | "coqui" | "xtts" | "custom"
    tts_voice: str = "default"


@dataclass
class IndoVoiceConfig:
    """Top-level configuration for the full Indo-Voice-3 system."""

    model: ModelConfig = field(default_factory=ModelConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    ablation: AblationConfig = field(default_factory=AblationConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)

    @classmethod
    def from_env(cls) -> "IndoVoiceConfig":
        """Build config from environment variables (production entrypoint)."""
        import os

        def _int(key: str, default: int) -> int:
            return int(os.getenv(key, default))

        def _float(key: str, default: float) -> float:
            return float(os.getenv(key, default))

        def _bool(key: str, default: bool) -> bool:
            val = os.getenv(key)
            if val is None:
                return default
            return val.lower() in ("1", "true", "yes", "on")

        model = ModelConfig(
            vocab_size=_int("INDO_VOCAB_SIZE", 32000),
            d_model=_int("INDO_D_MODEL", 768),
            d_k=_int("INDO_D_K", 64),
            d_v=_int("INDO_D_V", 64),
            n_skills=_int("INDO_N_SKILLS", 8),
            top_k=_int("INDO_TOP_K", 2),
            k_max=_int("INDO_K_MAX", 4),
            dendrite_window=_int("INDO_DENDRITE_WINDOW", 8),
            max_seq_len=_int("INDO_MAX_SEQ_LEN", 2048),
            dropout=_float("INDO_DROPOUT", 0.1),
            tie_embeddings=_bool("INDO_TIE_EMBEDDINGS", True),
        )

        safety = SafetyConfig(
            truncate_write_window=_int("INDO_TRUNCATE_WINDOW", 32),
            gate_cap=_float("INDO_GATE_CAP", 0.95),
            gate_floor=_float("INDO_GATE_FLOOR", 0.05),
            gate_target=_float("INDO_GATE_TARGET", 0.35),
            max_memory_norm=_float("INDO_MAX_MEM_NORM", 10.0),
        )

        ablation = AblationConfig(
            use_metacognition=_bool("INDO_USE_META", True),
            use_neuromodulator=_bool("INDO_USE_NEUROMOD", True),
            use_fast_memory=_bool("INDO_USE_FAST_MEM", True),
            use_slow_memory=_bool("INDO_USE_SLOW_MEM", True),
        )

        audio = AudioConfig(
            sample_rate=_int("INDO_SAMPLE_RATE", 16000),
            n_mels=_int("INDO_N_MELS", 80),
            encoder_layers=_int("INDO_AUDIO_LAYERS", 6),
            encoder_dim=_int("INDO_AUDIO_DIM", 512),
            encoder_heads=_int("INDO_AUDIO_HEADS", 8),
        )

        voice = VoiceConfig(
            max_new_tokens=_int("INDO_MAX_NEW_TOKENS", 256),
            temperature=_float("INDO_TEMPERATURE", 0.7),
            top_p=_float("INDO_TOP_P", 0.9),
            streaming_chunk_size=_int("INDO_STREAM_CHUNK", 4),
            primary_language=os.getenv("INDO_PRIMARY_LANG", "hi"),
            enable_hinglish=_bool("INDO_HINGLISH", True),
            tts_provider=os.getenv("INDO_TTS_PROVIDER", "none"),
        )

        return cls(model=model, safety=safety, ablation=ablation, audio=audio, voice=voice)
