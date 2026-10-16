# Scaling SENTINEL / AERIS-64

## Single process (default)

One loaded model + one `_lock`. Concurrent chats **queue**.

| Knob | Env | Default |
|------|-----|---------|
| Max waiters + in-flight | `NEUROFIELD_MAX_QUEUE` | 32 |
| Metrics | `GET /metrics` | `sentinel_model_queue_*` |

When the queue is full, chat/completions return **503** `server_busy`.

## Horizontal (real parallel generate)

Torch weights are **per process**. Do **not** use `uvicorn --workers N` with one shared model object.

Pattern:

1. Start N servers: `CUDA_VISIBLE_DEVICES=i PORT=800i` each loads a checkpoint.
2. Load balancer (nginx/caddy) round-robin `/v1/chat`.
3. Sticky sessions optional (same `session_id` → same backend) for KV/state.

## What stays serial inside one worker

Fast Δ-memory and slow ring updates are sequential in time — that is the model design, not a server bug.
