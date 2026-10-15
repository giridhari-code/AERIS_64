# Frontier-style structure (enforced)

```text
Browser → frontend/ (:3000) → backend API (:8000) → packages/sentinel (model)
```

| Layer | Path | Port |
|-------|------|------|
| **Frontend** | `frontend/` | 3000 |
| **Backend** | `backend/` + `packages/sentinel/api` + `serving/` | 8000 |
| **Model** | `packages/sentinel/` | — |

```text
/
├── frontend/                 # Bun web client only
│   ├── public/
│   └── src/
├── backend/                  # API service entry (run.sh)
├── packages/
│   └── sentinel/             # Model library (train, modules, tools, API)
│       ├── api/              # uvicorn entry
│       ├── serving/          # FastAPI implementation
│       ├── model.py · modules/
│       ├── training/ · checkpoint/
│       └── megha/ · tools/ · safety/
├── configs/ · checkpoints/ · data/
├── scripts/ · tests/ · docs/ · deploy/
└── pyproject.toml            # packages.sentinel
```

## Rules

1. UI → `frontend/` only  
2. HTTP service scripts → `backend/`  
3. All Python model/API code → `packages/sentinel/`  
4. `PYTHONPATH=packages` (not `src`)  
5. No HTML under `packages/sentinel/serving/static/`  

## Run

```bash
export PYTHONPATH=packages
export NEUROFIELD_CHECKPOINT=checkpoints/examples/AERIS_main
uvicorn "sentinel.api.app:create_app" --factory --port 8000

cd frontend && bun run dev
```
