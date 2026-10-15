# SENTINEL production guide

## What “production” means here

| Layer | Production-ready via this guide? |
|-------|----------------------------------|
| API service (auth, headers, no demo fallback, ready probe) | Yes |
| Policy (Megha + Bhutu) | Yes — gate, not a full SOC |
| Model answer quality | **No** — depends on your data + training |

## Checklist

1. Train / evaluate your own checkpoint (not `checkpoints/examples/*` for public traffic).
2. Set:
   ```bash
   export NEUROFIELD_ENV=production
   export NEUROFIELD_API_KEYS='long-random-1,long-random-2'
   export NEUROFIELD_CHECKPOINT=/var/lib/sentinel/model
   export NEUROFIELD_DEVICE=cuda   # or cpu
   export NEUROFIELD_LOG_JSON=1
   export NEUROFIELD_METRICS_PUBLIC=0
   export NEUROFIELD_HOST=127.0.0.1   # put nginx/caddy in front
   ```
3. Run gate:
   ```bash
   PYTHONPATH=src python scripts/ops/production_check.py
   ```
4. Serve behind TLS reverse proxy; do not expose bare `:8000` to the internet.
5. Monitor `/readyz`, `/metrics` (with key), and application logs.

## Explicit product limits

- Research / company architecture demo is fine with disclaimers.
- Do not market as a finished general frontier assistant until eval quality passes your bar.
- Megha is defensive assistance only; no unauthorized offensive use.
