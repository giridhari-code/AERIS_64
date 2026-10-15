# SENTINEL / AERIS_64 — folder roles (frontend · backend · model)

**Problem this fixes:** UI HTML, TypeScript edge, and Python model code were easy to mix.
Use **one place per role**. Do not put model weights or training code under `apps/web`.

---

## Three layers (strict)

| Layer | Path | Port | Responsibility |
|-------|------|------|----------------|
| **Frontend** | `apps/web/` | **:3000** | Chat UI, product pages, Buddy UI, proxy to Python |
| **Backend (API)** | `src/sentinel/serving/` | **:8000** | JSON API only (`/v1/chat`, tools, health). **No primary UI** |
| **Model core** | `src/sentinel/` (modules, model, checkpoint, training, …) | — | Neural field, memory, train, load weights |

```text
Browser  →  :3000 apps/web (Bun)  →  proxy  →  :8000 Python API  →  model weights
```

---

## What goes where

### Frontend only — `apps/web/`

```text
apps/web/
├── public/           # HTML/CSS/assets served by Bun (SOURCE OF TRUTH for UI)
│   ├── index.html    # chat UI
│   ├── product.html  # Kalofacts / AjeebWork product shell
│   └── logo.svg
├── src/
│   ├── server.ts     # Bun edge server
│   ├── proxy.ts      # forwards /v1/* to Python :8000
│   ├── buddy.ts      # Buddy float / navbar
│   └── git.ts
├── package.json
└── README.md
```

- Edit UI here only.
- Run: `cd apps/web && bun install && bun run dev`

### Backend API only — `src/sentinel/serving/`

```text
src/sentinel/serving/
├── server.py         # FastAPI app factory
├── auth.py, user_memory.py, …
└── static/           # LEGACY optional copy of HTML (see below)
```

- Default: **API JSON only** (`SERVE_UI=0`).
- Do **not** add new product UI here; add under `apps/web/public/`.

### Model only — rest of `src/sentinel/`

```text
src/sentinel/
├── model.py, config.py, modules/   # architecture
├── checkpoint/                     # load/save weights
├── training/, data/, eval/         # train / data / metrics
├── rag/, tools/, megha/, safety/   # retrieval, tools, defense
├── buddy/, kalofacts/, ajeebwork/  # product features (server-side)
└── sandbox/, sdk/, tokenizer/
```

Weights live under `checkpoints/` (examples in git are side-cars only; large `.safetensors` stay local / not mixed into frontend).

---

## No dual HTML (fixed in tree)

| Path | Role |
|------|------|
| **`apps/web/public/*.html`** | **Only** UI assets (chat + product) |
| **`src/sentinel/serving/static/`** | **No HTML** — README only; duplicates removed |

If `SERVE_UI=1` for local debug, Python serves **`apps/web/public`** (same files). Default backend is API-only:

```bash
export SERVE_UI=0
```

---

## Removed / do not revive

| Path | Status |
|------|--------|
| `src/neurofield/` | **Retired** — package renamed to `sentinel`. Stale `__pycache__` should be deleted. |
| Mixing model files into `apps/web/` | **Forbidden** |

Import style:

```python
# correct
from sentinel.model import SENTINEL
from sentinel.serving.server import create_app

# wrong (old)
from neurofield...
```

```bash
uvicorn "sentinel.serving.server:create_app" --factory --port 8000
```

---

## Quick run (clean split)

```bash
# Terminal 1 — model backend (JSON only)
cd /path/to/AERIS_64   # or SENTINEL repo root
export PYTHONPATH=src
export NEUROFIELD_CHECKPOINT=checkpoints/examples/AERIS_main   # or your frontier ckpt
uvicorn "sentinel.serving.server:create_app" --factory --host 0.0.0.0 --port 8000

# Terminal 2 — frontend
cd apps/web
bun install && bun run dev
# open http://localhost:3000
```

---

## Checklist before commit

1. UI change → only under `apps/web/`
2. API / auth / session → `src/sentinel/serving/`
3. Model math / train → `src/sentinel/modules|training|model.py`
4. No new files under `src/neurofield/`
5. If you must update legacy static, also update `apps/web/public/` (or delete legacy path later)
