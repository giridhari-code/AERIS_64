"""Build production Voice stack from settings + checkpoint."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Tuple

import torch

from sentinel.checkpoint.resolve import load_model_from_checkpoint
from sentinel.voice_config import IndoVoiceConfig, AudioConfig, ModelConfig, VoiceConfig
from sentinel.voice_settings import Settings, get_settings
from sentinel.modules.audio_encoder import AudioEncoder
from sentinel.modules.modality_projector import ModalityProjector
from sentinel.voice.adapter import VoiceAdapter
from sentinel.voice.stt_tts import STTBackend, TTSBackend, create_stt, create_tts

logger = logging.getLogger("indo_voice.factory")


class TokenizerBridge:
    """Uniform encode/decode interface over Char / BPE / HF tokenizers."""

    def __init__(self, tokenizer: Any):
        self._tok = tokenizer
        self.eos_token_id = getattr(tokenizer, "eos_token_id", None)
        stops = getattr(tokenizer, "stop_ids", None) or set()
        if self.eos_token_id is None and stops:
            self.eos_token_id = next(iter(stops))

    @property
    def vocab_size(self) -> int:
        return int(getattr(self._tok, "vocab_size", len(self._tok)))

    def encode(self, text: str, return_tensors: Optional[str] = None, **kwargs):
        if hasattr(self._tok, "encode"):
            ids = self._tok.encode(text)
        else:
            raise TypeError("Tokenizer has no encode()")
        if return_tensors == "pt":
            return torch.tensor([ids] if not isinstance(ids[0], list) else ids, dtype=torch.long)
        return ids

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.view(-1).tolist()
        if isinstance(ids, int):
            ids = [ids]
        return self._tok.decode(ids)


def build_voice_system(
    settings: Optional[Settings] = None,
    checkpoint: Optional[str] = None,
    require_checkpoint: bool = True,
) -> Tuple[VoiceAdapter, Any, TokenizerBridge, STTBackend, TTSBackend, Settings]:
    """
    Construct a production-ready voice system.

    Returns:
        adapter, sentinel_model, tokenizer_bridge, stt, tts, settings
    """
    settings = settings or get_settings()
    ckpt = checkpoint or settings.checkpoint_dir
    ckpt_path = Path(ckpt)

    if require_checkpoint and not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. "
            "Set INDO_CHECKPOINT to a folder with model.safetensors + config.json + vocab."
        )

    model = None
    tok_bridge = None
    d_model = 768
    vocab_size = 32000

    if ckpt_path.exists():
        model, tokenizer, full_cfg, meta = load_model_from_checkpoint(
            str(ckpt_path), device=settings.device, strict=False
        )
        d_model = full_cfg.model.d_model
        vocab_size = full_cfg.model.vocab_size
        tok_bridge = TokenizerBridge(tokenizer)
        logger.info(
            "Loaded checkpoint %s d_model=%s vocab=%s params=%s",
            ckpt_path,
            d_model,
            vocab_size,
            sum(p.numel() for p in model.parameters()),
        )
    elif require_checkpoint:
        raise FileNotFoundError(f"Checkpoint required: {ckpt_path}")

    indo_cfg = IndoVoiceConfig.from_env()
    indo_cfg.model.d_model = d_model
    indo_cfg.model.vocab_size = vocab_size
    indo_cfg.audio.sample_rate = settings.sample_rate
    indo_cfg.voice.max_new_tokens = settings.max_new_tokens
    indo_cfg.voice.temperature = settings.temperature
    indo_cfg.voice.top_p = settings.top_p
    indo_cfg.voice.primary_language = settings.primary_language
    indo_cfg.voice.enable_hinglish = settings.enable_hinglish
    indo_cfg.voice.tts_provider = settings.tts_provider

    adapter = VoiceAdapter(indo_cfg)
    if model is not None:
        adapter.set_sentinel(model)
    if tok_bridge is not None:
        adapter.set_tokenizer(tok_bridge)

    adapter.audio_encoder = AudioEncoder(indo_cfg.audio)
    adapter.projector = ModalityProjector(indo_cfg.audio, indo_cfg.model)

    if settings.audio_weights:
        aw = Path(settings.audio_weights)
        if aw.is_file():
            blob = torch.load(str(aw), map_location=settings.device, weights_only=True)
            if "encoder" in blob:
                adapter.audio_encoder.load_state_dict(blob["encoder"], strict=False)
            if "projector" in blob:
                adapter.projector.load_state_dict(blob["projector"], strict=False)
            logger.info("Loaded audio weights from %s", aw)

    adapter.to(settings.device)
    adapter.eval()

    stt = create_stt(
        settings.stt_provider,
        model_size=settings.whisper_size,
        device=settings.stt_device,
    )
    tts = create_tts(settings.tts_provider, model_name=settings.tts_model)

    return adapter, model, tok_bridge, stt, tts, settings
