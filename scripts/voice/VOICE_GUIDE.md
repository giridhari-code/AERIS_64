# Voice (Indo-Voice-3.0 integrated)

Speech stack from [indo-voice-3.0](https://github.com/giridhari-code/indo-voice-3.0) is **in this repo**:

```text
packages/sentinel/voice/           # CLI, server, STT/TTS, adapter, train_audio
packages/sentinel/modules/
  audio_encoder.py
  modality_projector.py
  hindi_router.py
```

## Install

```bash
pip install -e ".[voice]"
# optional production STT:
# pip install faster-whisper
```

## After 109M (or any) LM train

```bash
export PYTHONPATH=packages
export INDO_CHECKPOINT=$PWD/runs/aeris_109m_pretrain/final   # model.safetensors folder
export INDO_DEVICE=cuda

python -m sentinel.voice.cli health
python -m sentinel.voice.cli generate --text "Namaste"
python -m sentinel.voice.cli serve --host 0.0.0.0 --port 8000
```

## Train audio encoder (Hindi)

```bash
# manifest.jsonl lines: {"wav": "path.wav", "text": "..."}
python -m sentinel.voice.train_audio \
  --data data/hindi_speech/manifest.jsonl \
  --ckpt $INDO_CHECKPOINT \
  --steps 2000 \
  --output checkpoints/indo_audio.pt

export INDO_AUDIO_WEIGHTS=checkpoints/indo_audio.pt
```

## Architecture

```text
Audio → STT (optional) → text → SENTINEL
Audio → AudioEncoder → Projector → SENTINEL(inputs_embeds)
SENTINEL → tokens → TTS → audio
```

Env template: `.env.example.voice`
