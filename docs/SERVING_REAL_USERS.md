# Serving real users (not demo)

Engine path is ready for **trusted multi-user HTTP** behind a reverse proxy.  
Model *quality* still depends on your checkpoint + SFT — this doc is the **service** bar.

## Preconditions

1. Checkpoint folder with `model.safetensors` + tokenizer  
2. J-space session isolation (rule mode) — fixed in engine (`state["jspace"]`)  
3. `pip install -e ".[serve]"`

## Production env

```bash
export NEUROFIELD_ENV=production
export NEUROFIELD_CHECKPOINT=/path/to/final
export NEUROFIELD_DEVICE=cuda   # or cpu
export NEUROFIELD_DTYPE=bfloat16
export NEUROFIELD_API_KEYS="key1,key2"
export NEUROFIELD_RATE_LIMIT=60
export NEUROFIELD_RATE_WINDOW=60
export NEUROFIELD_MAX_INFLIGHT=4
export NEUROFIELD_CORS_ORIGINS="https://your.app"
# export NEUROFIELD_EXPOSE_DOCS=1   # only if you want /docs
# export NEUROFIELD_TRUST_PROXY=1   # only behind trusted reverse proxy

export PYTHONPATH=packages
uvicorn sentinel.api.app:create_app --factory --host 0.0.0.0 --port 8000 --workers 1
```

**Workers = 1** recommended: model + lock are process-local. Scale with multiple machines or a queue, not many workers sharing one GPU carelessly.

## Client

```bash
curl -s $HOST/v1/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: key1" \
  -d '{"prompt":"hello","max_new_tokens":64}'
```

## What production mode enforces

| Control | Behavior |
|---------|----------|
| API keys | Required (`X-API-Key`) |
| Demo checkpoint | Disabled |
| Rate limit | Per key/IP sliding window |
| Max in-flight | Semaphore; 503 when busy |
| OpenAPI | Off unless `NEUROFIELD_EXPOSE_DOCS=1` |
| Security headers | nosniff, DENY frame, no-store |
| Session isolation | New Sequence per request; J-space in state |

## Honest limits

| Not yet | Meaning |
|---------|---------|
| Continuous batching | One generate under lock at a time |
| Horizontal autoscaling story | You run N single-worker replicas + LB |
| Per-user long memory across requests | `session_id` is echo-only today |
| Content moderation | Add at proxy or filter layer |
| Frontier quality | Train/SFT your weights |

## Checklist before public traffic

- [ ] Strong SFT checkpoint evaluated offline  
- [ ] TLS at reverse proxy (nginx/caddy)  
- [ ] API keys rotated, not in git  
- [ ] `/readyz` in load balancer  
- [ ] Rate limits set for your GPU  
- [ ] Logs without storing full prompts if required by policy  

## Dev vs prod

| | Dev | Production |
|--|-----|------------|
| `NEUROFIELD_ENV` | unset | `production` |
| Keys | optional | required |
| Demo ckpt | allowed | forbidden |
| Docs | on | off by default |
