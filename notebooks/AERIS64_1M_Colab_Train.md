# AERIS_64 — 1M corpus training (Google Colab)

GPU runtime recommended (T4 / L4 / A100).

## 1. Clone + install

```python
# @title Install
!git clone https://github.com/giridhari-code/AERIS_64.git
%cd AERIS_64
!pip -q install -e ".[train]" || pip -q install -e .
!pip -q install torch --index-url https://download.pytorch.org/whl/cu124
```

## 2. Build ~1M dataset (Wikipedia + Gutenberg + literature, no code)

```python
# @title Build 1M pack
!PYTHONPATH=packages python scripts/data/build_1m_corpus.py
!ls -lh artifacts/dataset_1m/
!python -c "import json; print(json.load(open('artifacts/dataset_1m/summary.json'))['total_tokens'])"
```

## 3. Train SENTINEL (scratch small, or resume checkpoint)

### From scratch (small preset)

```python
# @title Train 1M
import os
os.environ["PYTHONPATH"] = "packages"
!python scripts/train/train_company.py \
  --data artifacts/dataset_1m/aeris64_1m_train.jsonl \
  --out checkpoints/aeris64_1m_ft \
  --steps 1200 \
  --preset small \
  --device cuda \
  --grad-accum 4 \
  --warmup 80 \
  --cosine \
  --lr 5e-4 \
  --product AERIS_64
```

### Or resume from example weights

```python
!python scripts/train/train_finetune.py \
  --resume checkpoints/examples/AERIS_main \
  --data artifacts/dataset_1m/aeris64_1m_train.jsonl \
  --out checkpoints/aeris64_1m_ft \
  --steps 800 \
  --preset small \
  --device cuda \
  --grad-accum 4
```

### Shell helper

```bash
DEVICE=cuda STEPS=1200 bash scripts/train/train_1m_colab.sh
```

## 4. Serve checkpoint

```python
import os
os.environ["NEUROFIELD_CHECKPOINT"] = "checkpoints/aeris64_1m_ft"
os.environ["NEUROFIELD_DEVICE"] = "cuda"
!PYTHONPATH=packages python -m uvicorn sentinel.api.app:create_app --factory --host 0.0.0.0 --port 8000
```

## Notes

| Item | Value |
|------|--------|
| Data | `aeris64_1m_train.jsonl` (`text` field) |
| Engine | SENTINEL |
| Typical Colab T4 | `small` preset, batch via `--grad-accum` |
| Longer quality | raise `--steps` (2000–5000) if time allows |

Token / private repo: use a PAT in `git clone https://USER:TOKEN@github.com/...` then revoke after.
