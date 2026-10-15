# Backend — inference API

Frontier-style service layer. **HTTP API only** (no browser UI).

| | |
|--|--|
| **Port** | `8000` |
| **Code** | `src/sentinel/serving/` (package) · entry `src/sentinel/api/` |
| **UI** | lives in `/frontend` on port `3000` |

```bash
export PYTHONPATH=src
export NEUROFIELD_CHECKPOINT=checkpoints/examples/AERIS_main
uvicorn "sentinel.api.app:create_app" --factory --host 0.0.0.0 --port 8000
# equivalent: uvicorn "sentinel.serving.server:create_app" --factory --port 8000
```

Do not put HTML or Bun code here.
