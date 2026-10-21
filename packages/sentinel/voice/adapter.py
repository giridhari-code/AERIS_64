"""Indo-Voice-3 Voice Adapter.

High-level speech-to-speech interface around the SENTINEL core.

Responsibilities:
- Accept raw audio waveform
- Encode → project → run SENTINEL
- Stream text tokens (and optionally audio later)
- Maintain per-session two-speed memory state
- Hindi / Hinglish language routing
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from sentinel.voice_config import IndoVoiceConfig
from sentinel.modules.audio_encoder import AudioEncoder
from sentinel.modules.modality_projector import ModalityProjector
from sentinel.modules.hindi_router import HindiSkillRouter

logger = logging.getLogger("indo_voice.adapter")


@dataclass
class VoiceSession:
    """Per-user / per-call session state."""

    session_id: str
    state: Optional[Dict[str, Any]] = None
    turn_count: int = 0
    language_id: int = 1          # 0=neutral, 1=Hindi, 2=Hinglish
    history_tokens: List[int] = field(default_factory=list)
    slow_memory: Optional[Dict[str, Tensor]] = None


class VoiceAdapter(nn.Module):
    """
    Production-ready speech interface for Indo-Voice-3.

    Usage (training / offline):
        adapter = VoiceAdapter(cfg)
        logits, new_state = adapter(waveform, input_ids=..., state=...)

    Usage (realtime streaming):
        async for token_id, text_piece in adapter.stream_generate(waveform, session):
            ...
    """

    def __init__(
        self,
        cfg: IndoVoiceConfig,
        sentinel_model: Optional[nn.Module] = None,
        tokenizer: Optional[Any] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer

        # --- Voice stack ---
        self.audio_encoder = AudioEncoder(cfg.audio)
        self.projector = ModalityProjector(cfg.audio, cfg.model)
        self.hindi_router = HindiSkillRouter(cfg.model, cfg.voice)

        # Core language model (injected or created later)
        self.sentinel = sentinel_model

        # Simple language detector head (very lightweight)
        self.lang_detector = nn.Sequential(
            nn.Linear(cfg.audio.encoder_dim, 128),
            nn.GELU(),
            nn.Linear(128, 3),  # neutral / Hindi / Hinglish
        )

    def set_sentinel(self, model: nn.Module) -> None:
        """Attach the core SENTINEL model after construction."""
        self.sentinel = model

    def set_tokenizer(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    # ------------------------------------------------------------------
    # Forward (training / offline)
    # ------------------------------------------------------------------
    def forward(
        self,
        waveform: Optional[Tensor] = None,
        input_ids: Optional[Tensor] = None,
        targets: Optional[Tensor] = None,
        state: Optional[Dict] = None,
        language_id: Optional[Tensor] = None,
        **kwargs,
    ):
        """
        Multimodal forward with full audio injection support.

        Priority:
          1. waveform → AudioEncoder → Projector → inputs_embeds
          2. input_ids → normal text path
          3. Both → audio embeds used as prefix (caller can concat externally)
        """
        if self.sentinel is None:
            raise RuntimeError("SENTINEL model not attached. Call set_sentinel() first.")

        device = next(self.parameters()).device
        inputs_embeds = None

        if waveform is not None:
            features, _ = self.audio_encoder(waveform.to(device))
            inputs_embeds = self.projector(features)  # (B, T_a, d_model)

            # Language detection from mean-pooled audio
            pooled = features.mean(dim=1)
            lang_logits = self.lang_detector(pooled)
            if language_id is None:
                language_id = lang_logits.argmax(dim=-1)

        if inputs_embeds is not None and input_ids is None:
            # Pure audio-conditioned path
            out = self.sentinel(
                inputs_embeds=inputs_embeds,
                targets=targets.to(device) if targets is not None else None,
                state=state,
                **kwargs,
            )
        elif input_ids is not None:
            out = self.sentinel(
                input_ids=input_ids.to(device),
                targets=targets.to(device) if targets is not None else None,
                state=state,
                **kwargs,
            )
        else:
            raise ValueError("Provide waveform and/or input_ids")

        return out


    # ------------------------------------------------------------------
    # Realtime streaming generation
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def generate_stream(
        self,
        input_ids: Tensor,
        session: VoiceSession,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ):
        """
        Synchronous token streaming generator.

        Yields:
            token_id (int)
        """
        if self.sentinel is None:
            raise RuntimeError("SENTINEL model not attached.")

        device = next(self.parameters()).device
        max_new = max_new_tokens or self.cfg.voice.max_new_tokens
        temp = temperature if temperature is not None else self.cfg.voice.temperature
        top_p_val = top_p if top_p is not None else self.cfg.voice.top_p

        state = session.state
        if state is None and session.slow_memory is not None:
            state = self.sentinel.new_state(1, device, slow=session.slow_memory)

        generated: List[int] = []
        cur_ids = input_ids.to(device)

        for step in range(max_new):
            out = self.sentinel(
                input_ids=cur_ids,
                state=state,
                write_slow=True,
                collect_stats=False,
            )
            state = out.state
            logits = out.logits[:, -1, :]  # (1, vocab)

            # Temperature + top-p sampling
            if temp > 0:
                logits = logits / max(temp, 1e-6)
                probs = torch.softmax(logits, dim=-1)
                if top_p_val < 1.0:
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                    cum = torch.cumsum(sorted_probs, dim=-1)
                    mask = cum > top_p_val
                    mask[..., 0] = False
                    sorted_probs = sorted_probs.masked_fill(mask, 0.0)
                    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
                    next_idx = torch.multinomial(sorted_probs, 1)
                    next_token = sorted_idx.gather(-1, next_idx)
                else:
                    next_token = torch.multinomial(probs, 1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            token_id = int(next_token.item())
            generated.append(token_id)
            yield token_id

            # Prepare next step (T=1)
            cur_ids = next_token

            # Optional early stop (EOS) – caller should handle tokenizer.eos_token_id
            if self.tokenizer is not None and hasattr(self.tokenizer, "eos_token_id"):
                if token_id == self.tokenizer.eos_token_id:
                    break

        # Update session
        session.state = state
        session.turn_count += 1
        session.history_tokens.extend(generated)

        # Periodic slow-memory consolidation
        if (
            session.turn_count % self.cfg.voice.consolidate_every_turns == 0
            and hasattr(self.sentinel, "consolidate_turn")
        ):
            session.state = self.sentinel.consolidate_turn(session.state)
            if hasattr(self.sentinel, "export_slow_state"):
                session.slow_memory = self.sentinel.export_slow_state(session.state)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def create_session(self, session_id: str, language: str = "hi") -> VoiceSession:
        lang_map = {"neutral": 0, "hi": 1, "hinglish": 2, "en": 0}
        language_id = lang_map.get(language.lower(), 1)
        return VoiceSession(session_id=session_id, language_id=language_id)

    def param_report(self) -> Dict[str, int]:
        report = {}
        report.update(self.audio_encoder.param_report())
        report.update(self.projector.param_report())
        report.update(self.hindi_router.param_report())
        report["lang_detector"] = sum(p.numel() for p in self.lang_detector.parameters())
        if self.sentinel is not None and hasattr(self.sentinel, "param_report"):
            report.update(self.sentinel.param_report())
        report["adapter_total"] = sum(p.numel() for p in self.parameters())
        return report
