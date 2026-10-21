"""STT + TTS for Indo-Voice-3 — config-driven, no hardcode."""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger("indo_voice.stt_tts")


@dataclass
class AudioChunk:
    samples: np.ndarray  # float32 mono [-1, 1]
    sample_rate: int
    duration_sec: float


# ----- STT -----

class STTBackend(ABC):
    @abstractmethod
    def transcribe(self, audio: AudioChunk, language: str = "hi") -> str:
        ...


class WhisperSTT(STTBackend):
    """faster-whisper (preferred) or openai-whisper."""

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        self.model_size = model_size or os.getenv("INDO_WHISPER_SIZE", "base")
        self.device = device or os.getenv("INDO_STT_DEVICE", "cpu")
        self.compute_type = compute_type
        self._model = None
        self._backend = None

    def _load(self):
        if self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type if self.device == "cpu" else "float16",
            )
            self._backend = "faster_whisper"
            logger.info("STT: faster-whisper model=%s device=%s", self.model_size, self.device)
            return
        except ImportError:
            pass
        try:
            import whisper
            self._model = whisper.load_model(self.model_size, device=self.device)
            self._backend = "openai_whisper"
            logger.info("STT: openai-whisper model=%s", self.model_size)
            return
        except ImportError as e:
            raise RuntimeError(
                "Install STT: pip install faster-whisper   OR   pip install openai-whisper"
            ) from e

    def transcribe(self, audio: AudioChunk, language: str = "hi") -> str:
        self._load()
        lang = "hi" if language in ("hi", "hinglish", "hindi") else language
        if self._backend == "faster_whisper":
            segments, info = self._model.transcribe(
                audio.samples.astype(np.float32),
                language=lang,
                beam_size=1,
                vad_filter=True,
            )
            text = " ".join(s.text.strip() for s in segments).strip()
            logger.debug("whisper lang=%s text=%s", getattr(info, "language", lang), text[:80])
            return text
        # openai whisper
        result = self._model.transcribe(
            audio.samples.astype(np.float32),
            language=lang,
            fp16=(self.device != "cpu"),
        )
        return (result.get("text") or "").strip()


class DummySTT(STTBackend):
    def transcribe(self, audio: AudioChunk, language: str = "hi") -> str:
        logger.warning("DummySTT — empty transcript")
        return ""


# ----- TTS -----

class TTSBackend(ABC):
    @abstractmethod
    def synthesize(self, text: str, language: str = "hi", voice: str = "default") -> AudioChunk:
        ...


class CoquiTTS(TTSBackend):
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or os.getenv(
            "INDO_TTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2"
        )
        self._tts = None

    def _load(self):
        if self._tts is not None:
            return
        from TTS.api import TTS
        self._tts = TTS(self.model_name)
        logger.info("TTS: Coqui %s", self.model_name)

    def synthesize(self, text: str, language: str = "hi", voice: str = "default") -> AudioChunk:
        self._load()
        lang = "hi" if language in ("hi", "hinglish", "hindi") else language
        kwargs = {"text": text, "language": lang}
        # XTTS may need speaker_wav — fallback to model default
        try:
            wav = self._tts.tts(**kwargs)
        except TypeError:
            wav = self._tts.tts(text=text)
        samples = np.asarray(wav, dtype=np.float32)
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        sr = 22050
        synth = getattr(self._tts, "synthesizer", None)
        if synth is not None and hasattr(synth, "output_sample_rate"):
            sr = int(synth.output_sample_rate)
        return AudioChunk(samples=samples, sample_rate=sr, duration_sec=len(samples) / sr)


class DummyTTS(TTSBackend):
    def synthesize(self, text: str, language: str = "hi", voice: str = "default") -> AudioChunk:
        sr = 16000
        dur = max(0.4, min(8.0, len(text) * 0.07))
        t = np.linspace(0, dur, int(sr * dur), dtype=np.float32)
        # simple beep pattern so file is audible
        sig = 0.2 * np.sin(2 * np.pi * 440 * t) * np.exp(-t * 1.5)
        return AudioChunk(samples=sig.astype(np.float32), sample_rate=sr, duration_sec=dur)


def create_stt(provider: Optional[str] = None, **kwargs) -> STTBackend:
    provider = (provider or os.getenv("INDO_STT_PROVIDER", "auto")).lower()
    allow_dummy = os.getenv("INDO_ALLOW_DUMMY_IO", "0").lower() in ("1", "true", "yes")
    if provider in ("whisper", "faster-whisper", "auto"):
        try:
            return WhisperSTT(**kwargs)
        except Exception as e:
            if provider == "auto" and allow_dummy:
                logger.warning("Whisper unavailable (%s); INDO_ALLOW_DUMMY_IO set → DummySTT", e)
                return DummySTT()
            raise RuntimeError(
                f"STT backend unavailable: {e}. Install faster-whisper or set INDO_STT_PROVIDER=dummy "
                "with INDO_ALLOW_DUMMY_IO=1 for offline tests only."
            ) from e
    if provider == "dummy":
        if not allow_dummy:
            logger.warning("DummySTT requested — set INDO_ALLOW_DUMMY_IO=1 to silence this in CI")
        return DummySTT()
    raise ValueError(f"Unknown STT provider: {provider}")


def create_tts(provider: Optional[str] = None, **kwargs) -> TTSBackend:
    provider = (provider or os.getenv("INDO_TTS_PROVIDER", "auto")).lower()
    allow_dummy = os.getenv("INDO_ALLOW_DUMMY_IO", "0").lower() in ("1", "true", "yes")
    if provider in ("coqui", "xtts", "auto"):
        try:
            return CoquiTTS(**kwargs)
        except Exception as e:
            if provider == "auto" and allow_dummy:
                logger.warning("Coqui unavailable (%s); INDO_ALLOW_DUMMY_IO set → DummyTTS", e)
                return DummyTTS()
            if provider == "auto":
                logger.warning("Coqui unavailable (%s); using DummyTTS until TTS is installed", e)
                return DummyTTS()
            raise RuntimeError(f"TTS backend unavailable: {e}. pip install TTS") from e
    if provider == "dummy":
        return DummyTTS()
    raise ValueError(f"Unknown TTS provider: {provider}")


def waveform_to_tensor(chunk: AudioChunk, target_sr: int = 16000) -> Tensor:
    samples = chunk.samples.astype(np.float32)
    if chunk.sample_rate != target_sr:
        ratio = target_sr / float(chunk.sample_rate)
        new_len = max(1, int(len(samples) * ratio))
        x = np.linspace(0, len(samples) - 1, new_len)
        samples = np.interp(x, np.arange(len(samples)), samples).astype(np.float32)
    return torch.from_numpy(samples).unsqueeze(0)
