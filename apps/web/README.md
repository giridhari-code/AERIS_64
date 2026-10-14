# AERIS Web — TypeScript + Bun Frontend

| Port | Stack | Role |
|------|--------|------|
| **:3000** | Bun + TypeScript | **Frontend** — chat UI |
| **:8000** | Python | **Backend** — model only (JSON API) |

## Run both

```bash
# Terminal 1 — Python backend (no UI)
cd ~/AERIS_64
export PYTHONPATH=src
export NEUROFIELD_CHECKPOINT=checkpoints/examples/AERIS_main
uvicorn "neurofield.serving.server:create_app" --factory --host 0.0.0.0 --port 8000

# Terminal 2 — Bun frontend
cd ~/AERIS_64/apps/web
bun install && bun run dev
```

Open **http://localhost:3000** for chat.

- `:8000/` → JSON (backend info)
- `:3000/` → full chat UI

Legacy: `NEUROFIELD_SERVE_UI=1` serves the old HTML UI on Python port too.
