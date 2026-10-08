# NeuroField Company API Reference

Base URL: `http://127.0.0.1:8000`

## Auth

Optional. If `NEUROFIELD_API_KEYS=key1,key2` is set:

```
X-API-Key: key1
```

If unset, API is open (development mode).

## Rate limit

Default 120 requests / 60s per API key or IP.  
Override: `NEUROFIELD_RATE_LIMIT`, `NEUROFIELD_RATE_WINDOW`.

## Endpoints

### GET /healthz
Liveness.

### GET /readyz
Readiness; 503 if model not loaded.

### GET /metrics
Prometheus text metrics.

### GET /v1/models
List model ids.

### POST /v1/chat
```json
{
  "prompt": "namaste",
  "max_new_tokens": 48,
  "temperature": 0.4,
  "session_id": "optional"
}
```

### POST /v1/chat/stream
Same body; response is **SSE** (`text/event-stream`):

```
data: {"session_id":"...","event":"start"}
data: {"token":"N","id":12}
data: {"event":"done","latency_ms":123.4}
data: [DONE]
```

### POST /v1/completions
Token-id interface for low-level clients.

### POST /v1/reset?session_id=...
Clear fast-memory session state.

## Start server

```bash
export CHECKPOINT=docs/neurofield_company
export NEUROFIELD_API_KEYS=dev-secret   # optional
PYTHONPATH=src uvicorn "neurofield.serving.server:create_app" --factory --host 0.0.0.0 --port 8000
```
