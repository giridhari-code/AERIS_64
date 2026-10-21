"""Indo-Voice speech stack (integrated from indo-voice-3.0).

Import submodules explicitly to avoid circular imports with modules.*:

    from sentinel.voice.factory import build_voice_system
    from sentinel.voice.cli import main
"""

__all__ = ["VoiceAdapter", "VoiceSession", "build_voice_system", "create_stt", "create_tts"]


def __getattr__(name: str):
    if name in ("VoiceAdapter", "VoiceSession"):
        from sentinel.voice.adapter import VoiceAdapter, VoiceSession
        return {"VoiceAdapter": VoiceAdapter, "VoiceSession": VoiceSession}[name]
    if name == "build_voice_system":
        from sentinel.voice.factory import build_voice_system
        return build_voice_system
    if name in ("create_stt", "create_tts", "AudioChunk"):
        from sentinel.voice import stt_tts
        return getattr(stt_tts, name)
    raise AttributeError(name)
