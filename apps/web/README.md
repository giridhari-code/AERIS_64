# SENTINEL Web — TypeScript + Bun **Frontend**

Strict split (see also [docs/STRUCTURE.md](../../docs/STRUCTURE.md)):

| Port | Stack | Role |
|------|--------|------|
| **:3000** | Bun + TypeScript | **Frontend** — chat UI, product, Buddy |
| **:8000** | Python (`src/sentinel`) | **Backend** — model JSON API only |

Do **not** put model weights, training, or FastAPI routes inside this folder.

## Run both

```bash
# Terminal 1 — Python backend (API only, no UI)
cd /path/to/repo
export PYTHONPATH=src
export NEUROFIELD_CHECKPOINT=checkpoints/examples/AERIS_main
uvicorn "sentinel.serving.server:create_app" --factory --host 0.0.0.0 --port 8000

# Terminal 2 — Bun frontend
cd apps/web
bun install && bun run dev
```

Open **http://localhost:3000** for chat.

- `:8000/` → JSON (backend info / health)
- `:3000/` → full chat UI (`public/index.html`)
- `:3000/product` → product shell (`public/product.html`)

Legacy only: `NEUROFIELD_SERVE_UI=1` serves a **copy** of HTML from `src/sentinel/serving/static/` on the Python port. Prefer Bun. Canonical UI files are under **`apps/web/public/`**.

## Layout

```text
apps/web/
├── public/          # SOURCE OF TRUTH for HTML/assets
│   ├── index.html
│   ├── product.html
│   └── logo.svg
└── src/
    ├── server.ts    # edge server
    ├── proxy.ts     # → Python :8000
    ├── buddy.ts
    └── git.ts
```

## Product UX

- Chat: `http://localhost:3000/`
- Product (Kalofacts / AjeebWork / SI_v1): `http://localhost:3000/product`
- All `/v1/*` APIs proxied to Python `:8000`
