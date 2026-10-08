# India language + Hindlish + spelling support

## What was added

1. **Dataset** `data/india_multilang.txt`
   - English
   - Hindlish (Roman Hindi+English)
   - Hindi Devanagari
   - Bengali samples
   - Tamil / Telugu / Marathi / Gujarati transliteration
   - **Spelling / missing-letter** variants (`namste`, `plz`, `modle`, …)

2. **Inference normalizer** `neurofield.utils.text_norm.normalize_prompt`
   - Fixes common typos before the model sees the text
   - Wired into `POST /v1/chat`

3. **Train script** `scripts/train_multilang.py`
   - Trains on multilang + optional `skills_real.txt`
   - Saves **safetensors folder** checkpoint

## Train

```bash
cd neurofield_v2_clean
pip install -e ".[serve]"

PYTHONPATH=src python scripts/train_multilang.py \
  --data data/india_multilang.txt \
  --extra data/skills_real.txt \
  --steps 1000 \
  --out docs/neurofield_india
```

## Serve

```bash
CHECKPOINT=docs/neurofield_india PYTHONPATH=src \
  uvicorn "neurofield.serving.server:create_app" --factory --host 0.0.0.0 --port 8000
```

## Limits (honest)

- Model is **character-level** and small — not a full multilingual LLM.
- Devanagari / Bangla quality needs **more data + more steps**.
- Best results: Hindlish + English + short Hindi phrases after training on this corpus.
- For production India apps, later upgrade to a subword tokenizer (BPE/SentencePiece) and larger model.

## Add more language data

Append lines to `data/india_multilang.txt` (any script UTF-8), then re-run `train_multilang.py`.
