# AERIS Web (TypeScript + Bun)

Edge layer for **AERIS_64**.

| Layer | Tech | Responsibility |
|-------|------|----------------|
| **Model** | Python | Training, neural field, fast/slow memory, inference |
| **Edge** | TypeScript + Bun | API gateway, git helpers, future agent orchestration / UI |

## Quick start

```bash
# 1. Start Python model server (from repo root)
export NEUROFIELD_CHECKPOINT=checkpoints/local_run   # or your ckpt
uvicorn "neurofield.serving.server:create_app" --factory --host 0.0.0.0 --port 8000

# 2. Start Bun edge
cd apps/web
bun install
bun run dev
# → http://localhost:3000
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service info |
| GET | `/healthz` | Web + Python health |
| POST | `/v1/chat` | Proxy to Python `/v1/chat` |
| GET | `/v1/git/status` | `git status` |
| GET | `/v1/git/log` | recent commits |
| GET | `/v1/git/diff` | `git diff --stat` |

Env:

- `AERIS_PYTHON_URL` — Python backend (default `http://127.0.0.1:8000`)
- `PORT` — Bun listen port (default `3000`)
- `AERIS_REPO_ROOT` — git working tree (default cwd)
