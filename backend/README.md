# Backend — SENTINEL inference API

| | |
|--|--|
| **Port** | `8000` |
| **Package** | `packages/sentinel` |
| **Entry** | `uvicorn sentinel.api.app:create_app --factory` |

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

No HTML in this layer — UI is `frontend/` on port 3000.
