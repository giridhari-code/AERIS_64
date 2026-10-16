# SENTINEL Company API Reference

Base URL: `http://127.0.0.1:8000`

## Auth

Optional. If `NEUROFIELD_API_KEYS=key1,key2` is set:

```
X-API-Key: key1
```

If unset, API is open (development mode).

### Accounts (web client login/signup)

Optional. Accounts are stored server-side (SQLite, default
`data/auth/accounts.db`); passwords are scrypt-hashed, sessions are opaque
bearer tokens (only SHA-256 stored, 30-day TTL).

```
Authorization: Bearer <token>
```

| Env | Effect |
| --- | --- |
| `NEUROFIELD_AUTH=0` | disable accounts entirely |
| `NEUROFIELD_AUTH_DB` | sqlite path (default `data/auth/accounts.db`) |
| `NEUROFIELD_AUTH_TTL_DAYS` | session lifetime (default 30) |
| `NEUROFIELD_AUTH_REQUIRED=1` | chat endpoints require an account token or X-API-Key |

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

### GET /v1/auth/status
Public capability probe (no secrets):
`{"accounts":true,"guest_ok":true,"required":false,"users":3}`.

### POST /v1/auth/signup
`{"name":"Asha","email":"asha@example.com","password":"…"}` → 201
`{"token":"…","user":{id,name,email}}`. 409 when the email exists, 422 on
weak input.

### POST /v1/auth/login
`{"email":"…","password":"…"}` → `{"token":"…","user":{…}}`; 401 on bad
credentials.

### GET /v1/auth/me
`Authorization: Bearer` → `{"user":{…}}`; 401 when expired/revoked.

### POST /v1/auth/logout
`Authorization: Bearer` → `{"status":"bye"}` (revokes that one session).

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

### GET /v1/buddy?user_id=…
Decorative ASCII companion + egg state for that identity.

### POST /v1/buddy
`{"command":"/buddy|/buddy egg|/buddy pet","user_id":"…"}` →
`{"text":…,"buddy":{…},"egg":{…}}`. While the egg is unhatched, `/buddy`
answers with incubation status instead of the reveal.

### POST /v1/buddy/step
`{"user_id":"…","n":1}` → one incubation step (the chat UI calls this after
each user message):

```json
{"hatched":false,"egg":{"messages":3,"progress":5,"required":8,"hatched":false,
 "period_s":300,"next_step_in_s":240,"frames":["…"],"crack_frames":["…"]}}
```

Once progress reaches the requirement the response flips to `"hatched":true`
and carries the revealed `buddy`. Egg env:

| Env | Effect |
| --- | --- |
| `NEUROFIELD_BUDDY_DB` | egg store JSON (default `data/buddy/eggs.json`) |
| `NEUROFIELD_BUDDY_EGG_PERIOD` | passive seconds per step (default 300 = 5 min) |

## Start server

```bash
export CHECKPOINT=docs/neurofield_company
export NEUROFIELD_API_KEYS=dev-secret   # optional
PYTHONPATH=src uvicorn "neurofield.serving.server:create_app" --factory --host 0.0.0.0 --port 8000
```
