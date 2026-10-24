# Backend — SENTINEL inference API (core engine)

| | |
|--|--|
| **Port** | `8000` |
| **Package** | `packages/sentinel` |
| **Entry** | `uvicorn sentinel.api.app:create_app --factory` |
| **Language** | Python (PyTorch model) |

```bash
export NEUROFIELD_CHECKPOINT=/path/to/ckpt   # folder with model.safetensors
./backend/run.sh
# or: make backend
```

Env (optional long-context):

```bash
export NEUROFIELD_MAX_TURNS=24
export NEUROFIELD_MAX_CONTEXT=2048
```

## 0zcode Rust edge (optional)

The **core engine stays Python**. For a Rust-facing edge that proxies to this server:

```bash
# Terminal 1 — this core
./backend/run.sh

# Terminal 2 — Rust backend (std-only)
cd apps/0zcode && cargo run -- serve --port 8100 --core http://127.0.0.1:8000

# Point clients at the Rust edge
export ZEROZ_AERIS_URL=http://127.0.0.1:8100
```

See `apps/0zcode/src/backend.rs` and `0zcode serve --help`.

No HTML in this layer — UI is the 0zcode TUI.
