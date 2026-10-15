# SENTINEL Frontend — Bun + TypeScript

**Only UI lives here.** Model and training stay in `src/sentinel/`.

| Port | Stack | Role |
|------|--------|------|
| **:3000** | Bun | Frontend — chat, product, Buddy |
| **:8000** | Python `src/sentinel/serving` | Backend — model JSON API |

## Run

```bash
# Terminal 1 — backend (API only)
export PYTHONPATH=src
export NEUROFIELD_CHECKPOINT=checkpoints/examples/AERIS_main
uvicorn "sentinel.serving.server:create_app" --factory --host 0.0.0.0 --port 8000

# Terminal 2 — frontend
cd apps/frontend
bun install && bun run dev
```

Open http://localhost:3000

```text
apps/frontend/
├── public/          # HTML assets (only place for UI files)
│   ├── index.html
│   ├── product.html
│   └── logo.svg
└── src/
    ├── server.ts    # edge
    ├── proxy.ts     # → :8000
    ├── buddy.ts
    └── git.ts
```

Do not put model weights or Python training code in this folder.
