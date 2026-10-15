# Frontier-style folder structure

Layout mirrors product AI stacks (web client + inference API + model core):

```text
Browser  →  frontend/ (:3000)  →  backend API (:8000)  →  model weights
```

| Layer | Path | Port | Like |
|-------|------|------|------|
| **Frontend** | `frontend/` | 3000 | ChatGPT/Claude/Grok web UI |
| **Backend** | `backend/` + `src/sentinel/api` + `serving/` | 8000 | Inference API |
| **Model** | `src/sentinel/` (model, modules, training, checkpoint) | — | Core LM |

## Frontend only — `frontend/`

- HTML/TS/Bun only
- Proxies `/v1/*` to Python
- No `.py` model code, no checkpoints

## Backend only — API

- Entry: `uvicorn sentinel.api.app:create_app --factory`
- Implementation: `src/sentinel/serving/`
- Default: **JSON only** (`SERVE_UI=0`)
- Optional debug: `SERVE_UI=1` serves `frontend/public` (same files, not a second copy)

## Model only — `src/sentinel/`

- `model.py`, `modules/`, `training/`, `checkpoint/`
- Tools, Megha, RAG, safety, sandbox
- Weights under `checkpoints/` (git keeps side-cars only)

## Rules

1. UI change → `frontend/` only  
2. API/auth/session → `src/sentinel/serving/` (+ `api/` entry)  
3. Architecture / train → `src/sentinel/` model modules  
4. Never put weights or train loops under `frontend/`  
5. Never put Bun/HTML under `src/sentinel/serving/static/` (README only)

## Run

```bash
# Backend
export PYTHONPATH=src
export NEUROFIELD_CHECKPOINT=checkpoints/examples/AERIS_main
uvicorn "sentinel.api.app:create_app" --factory --port 8000

# Frontend
cd frontend && bun install && bun run dev
```
