> **Note (v2.2):** server env vars are now `NEUROFIELD_CHECKPOINT / NEUROFIELD_CONFIG / NEUROFIELD_DEVICE` (the short `CHECKPOINT/CONFIG/DEVICE` still work). For the 1B configuration see [TRAIN_1B.md](TRAIN_1B.md).

# NeuroField v2 — Run & Training Guide

Complete steps: install → train → serve → chat.

---

## 1. Setup (one time)

```bash
# unzip
unzip neurofield_v2_production.zip
cd neurofield_v2_clean

# Python 3.10+ recommended
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# install package + server deps
pip install -e ".[serve]"
```

Check:

```bash
PYTHONPATH=src python -c "from neurofield.model import NeuroField; print('OK')"
```

---

## 2. Run (inference / chat)

### Option A — India + Hindlish model (recommended)

```bash
cd neurofield_v2_clean
source .venv/bin/activate

CHECKPOINT=docs/neurofield_india PYTHONPATH=src \
  uvicorn "neurofield.serving.server:create_app" --factory --host 0.0.0.0 --port 8000
```

### Option B — Identity / code demo model

```bash
CHECKPOINT=docs/neurofield_trained PYTHONPATH=src \
  uvicorn "neurofield.serving.server:create_app" --factory --host 0.0.0.0 --port 8000
```

### Option C — Makefile

```bash
export CHECKPOINT=docs/neurofield_india
make serve-chat
```

### Open in browser

| URL | Purpose |
|-----|---------|
| http://127.0.0.1:8000/ | **Chat frontend** |
| http://127.0.0.1:8000/docs | Swagger API |
| http://127.0.0.1:8000/readyz | Model loaded? |

### API test (no UI)

```bash
curl -s http://127.0.0.1:8000/readyz

curl -s -X POST http://127.0.0.1:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt":"namaste","max_new_tokens":40,"temperature":0.3}'
```

### Common run errors

| Error | Fix |
|-------|-----|
| `model not loaded` | Set `CHECKPOINT=docs/neurofield_india` (folder path) |
| `No module named safetensors` | `pip install safetensors` |
| Port in use | Change `--port 8001` |
| Empty / garbage text | Train more steps or use matching vocab checkpoint |

**Important:** `CHECKPOINT` must point to a **folder** (safetensors) or a legacy `.pt` file.

```
docs/neurofield_india/          ← folder (preferred)
  model.safetensors
  config.json
  vocab.json
  meta.json
```

---

## 3. Training process

### 3.1 What you need

1. A **text file** dataset (UTF-8), one line or paragraph per idea  
2. Scripts under `scripts/`  
3. Enough time on CPU (GPU optional; code uses CPU by default)

### 3.2 Available datasets

| File | Content |
|------|---------|
| `data/india_multilang.txt` | English, Hindlish, Hindi, typos, more Indian langs |
| `data/skills_real.txt` | Self-learning, EQ, discipline skills text |

You can also create your own:

```bash
nano data/my_data.txt
# paste your text, save
```

### 3.3 Train India / Hindlish model

```bash
cd neurofield_v2_clean
source .venv/bin/activate

PYTHONPATH=src python scripts/train_multilang.py \
  --data data/india_multilang.txt \
  --extra data/skills_real.txt \
  --steps 1000 \
  --d-model 64 \
  --out docs/neurofield_india
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--data` | `data/india_multilang.txt` | Main corpus |
| `--extra` | `data/skills_real.txt` | Optional second file |
| `--steps` | 1000 | Training steps (more = better, slower) |
| `--d-model` | 64 | Model width |
| `--lr` | 3e-3 | Learning rate |
| `--out` | `docs/neurofield_india` | Output **folder** (safetensors) |

**Typical CPU time:** ~2–5 minutes for 500–1000 steps (small model).

### 3.4 Train skills-only model

```bash
PYTHONPATH=src python scripts/train_skills.py \
  --data data/skills_real.txt \
  --steps 800 \
  --out docs/neurofield_skills
```

### 3.5 What training does (pipeline)

```
1. Load text files (UTF-8)
2. Build character vocabulary (stoi / itos)
3. Create NeuroField model (dendrite, field, fast/slow memory, router…)
4. Loop:
     - sample random batches of character sequences
     - forward → next-char loss + error energy
     - backward → AdamW update
     - optional slow-memory replay
5. Save checkpoint folder:
     model.safetensors + config.json + vocab.json + meta.json
```

### 3.6 After training — run the new model

```bash
CHECKPOINT=docs/neurofield_india PYTHONPATH=src \
  uvicorn "neurofield.serving.server:create_app" --factory --host 0.0.0.0 --port 8000
```

Restart the server every time you change the checkpoint.

### 3.7 Add more language / spelling data

1. Edit `data/india_multilang.txt`  
2. Add lines, for example:

```text
kya kar rahe ho
kal chalo market
मुझे पानी चाहिए
speling mistakes are ok here
```

3. Re-train:

```bash
PYTHONPATH=src python scripts/train_multilang.py --steps 1500 --out docs/neurofield_india
```

### 3.8 Convert old `.pt` → safetensors folder

```bash
PYTHONPATH=src python -c "
from neurofield.checkpoint.io import convert_pt_to_safetensors
convert_pt_to_safetensors('docs/neurofield_trained.pt', 'docs/neurofield_trained')
print('done')
"
```

---

## 4. Quick checklist

```text
[ ] pip install -e ".[serve]"
[ ] CHECKPOINT=docs/neurofield_india
[ ] uvicorn ... --port 8000
[ ] open http://127.0.0.1:8000/
[ ] /readyz shows "ready"
[ ] chat: namaste / who am I / pls help
```

### Train more quality

```text
[ ] put better data in data/*.txt
[ ] train_multilang.py --steps 1500 or 2000
[ ] restart server with new CHECKPOINT folder
```

---

## 5. Related docs

| Doc | Topic |
|-----|--------|
| `docs/LANGUAGE_SUPPORT.md` | Hindlish / Hindi / typos |
| `docs/FIXES_AND_CHECKPOINT.md` | model not loaded, safetensors, multi-file |
| `docs/AUDIT_REPORT.md` | paper-faithful architecture fixes |
| `README.md` | project overview |

---

## 6. Docker (optional)

```bash
cd neurofield_v2_clean
# set checkpoint path inside compose/env as needed
docker compose -f deploy/docker/docker-compose.yml up --build
```

See `deploy/docker/` and `deploy/k8s/` for production-style deploy.


---

## 7. Company train / eval / stream (v2.1)

```bash
# Train company model (char tokenizer)
PYTHONPATH=src python scripts/train_company.py \
  --data data/company_corpus.txt data/india_multilang.txt data/skills_real.txt \
  --tokenizer char --steps 800 --d-model 64 --out docs/neurofield_company

# Optional BPE
PYTHONPATH=src python scripts/train_company.py --tokenizer bpe --bpe-vocab 400 --steps 800 --out docs/neurofield_company_bpe

# Eval
PYTHONPATH=src python scripts/eval_company.py --checkpoint docs/neurofield_company

# Serve with optional API key
export CHECKPOINT=docs/neurofield_company
export NEUROFIELD_API_KEYS=dev-secret
PYTHONPATH=src uvicorn "neurofield.serving.server:create_app" --factory --host 0.0.0.0 --port 8000
```

Streaming: `POST /v1/chat/stream` (SSE).


## Quality vs UI knobs

See **[QUALITY_AND_SCALING.md](QUALITY_AND_SCALING.md)**: quality = **data + size + align**; top models allow **much higher max_tokens**, AERIS should keep **~48**.


## Company-style full process

See [TRAINING_PROCESS_COMPANY_STYLE.md](TRAINING_PROCESS_COMPANY_STYLE.md).
