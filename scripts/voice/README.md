# Voice (in-repo Indo-Voice-3.0)

Integrated from https://github.com/giridhari-code/indo-voice-3.0

```bash
pip install -e ".[voice]"
export PYTHONPATH=packages
export INDO_CHECKPOINT=$PWD/runs/aeris_109m_pretrain/final

python -m sentinel.voice.cli health
python -m sentinel.voice.cli generate --text "Namaste"
python -m sentinel.voice.cli serve --host 0.0.0.0 --port 8000
```

Docs: [docs/guides/VOICE_INDO.md](../../docs/guides/VOICE_INDO.md)
