# AERIS Web (TypeScript + Bun) — Frontend

Real chat UI + API gateway. Python keeps the neural model.

| Port | Role |
|------|------|
| **:3000** (Bun) | Chat frontend + proxy + git helpers |
| **:8000** (Python) | Model inference only |

## Run

```bash
# Terminal 1 — Python model
cd ~/AERIS_64
export PYTHONPATH=src
export NEUROFIELD_CHECKPOINT=checkpoints/examples/AERIS_main
uvicorn "neurofield.serving.server:create_app" --factory --host 0.0.0.0 --port 8000

# Terminal 2 — Bun frontend
cd ~/AERIS_64/apps/web
bun install
bun run dev
```

Open **http://localhost:3000** — same chat UI as before, served by Bun.

## What Bun does

- Serves the chat UI (`public/index.html`)
- Proxies `/v1/chat`, `/readyz`, `/v1/reset`, … → Python `:8000`
- Local git helpers: `/v1/git/status|log|diff`
