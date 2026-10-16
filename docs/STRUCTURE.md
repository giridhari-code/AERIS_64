# SENTINEL — professional layout

```text
AERIS_64 / SENTINEL
├── packages/sentinel/     # Core engine + API (Python package)
│   ├── model.py           # SENTINEL forward (two-speed memory)
│   ├── modules/           # dendrite, field, memory, router, …
│   ├── serving/           # FastAPI routes, chat, tools
│   ├── api/app.py         # uvicorn entry: create_app
│   ├── training/          # train loops
│   ├── agent/             # SI_v0 / SI_v1
│   └── tools/             # calc, search, weather, …
├── backend/               # Inference process only (port 8000)
│   └── run.sh
├── frontend/              # Bun + TS (+ React) UI (port 3000)
├── configs/ data/ docs/ scripts/ tests/ deploy/
└── pyproject.toml         # package discover: packages/
```

## Run

```bash
# API
make backend
# or: PYTHONPATH=packages uvicorn sentinel.api.app:create_app --factory --port 8000

# UI
make frontend
# http://localhost:3000
```

## Long context

See `docs/guides/LONG_CONTEXT_MEMORY.md`.
