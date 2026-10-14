# API rate limits

## Python backend (:8000)

| Env | Default | Meaning |
|-----|---------|---------|
| NEUROFIELD_RATE_LIMIT | 60 | max requests per window |
| NEUROFIELD_RATE_WINDOW | 60 | window seconds |
| NEUROFIELD_TRUST_PROXY | 0 | if 1, use X-Forwarded-For |

All /v1/* routes limited via middleware. HTTP 429 + Retry-After.

## Bun frontend (:3000)

| Env | Default |
|-----|---------|
| AERIS_RATE_LIMIT | 60 |
| AERIS_RATE_WINDOW_MS | 60000 |

## Example

export NEUROFIELD_RATE_LIMIT=30
export NEUROFIELD_RATE_WINDOW=60
