# SDLC — NeuroField / AERIS

Production lifecycle for this repository. No demo stages: each gate is real.

## Stages

| Stage | Artifact | Gate |
|-------|----------|------|
| **1. Plan** | issue / design note | scope + non-goals written |
| **2. Build** | `src/neurofield/**` | `ruff` syntax rules + types where practical |
| **3. Unit / integration** | `tests/**` | `pytest` green on CI |
| **4. Train** | `checkpoints/<run>/model.safetensors` | train script completes; weights on disk |
| **5. Eval** | `eval_report.json` | `scripts/eval/eval_company.py` on held-out text |
| **6. Package** | wheel / docker image | `python -m build` or `deploy/docker` |
| **7. Release** | git tag `vX.Y.Z` | CHANGELOG entry + CI green on `main` |
| **8. Operate** | running server | `/healthz` + `/readyz`; optional API key |

## Local loop

```bash
pip install -e ".[dev,serve,sdk]"
ruff check src/ tests/ --select E9,F63,F7,F82
PYTHONPATH=src pytest tests/ -q
```

## Train → eval → serve

```bash
# data
cat data/raw/*.txt > data/ready/train.txt

# train
PYTHONPATH=src python scripts/train/train_company.py \
  --data data/ready/train.txt --out checkpoints/run1

# eval
PYTHONPATH=src python scripts/eval/eval_company.py \
  --checkpoint checkpoints/run1 --val-data data/ready/val.txt \
  --out checkpoints/run1/eval_report.json

# serve
NEUROFIELD_CHECKPOINT=checkpoints/run1 PYTHONPATH=src \
  uvicorn "neurofield.serving.server:create_app" --factory --host 0.0.0.0 --port 8000
```

## SDK (consumer apps)

```python
from neurofield.sdk import AerisClient

with AerisClient() as client:  # or AerisClient("https://your-host:8000")
    client.health()
    r = client.chat("Who are you?", max_new_tokens=64)
    print(r.text, r.latency_ms)
```

Env:

- `NEUROFIELD_BASE_URL` — server origin
- `NEUROFIELD_API_KEY` — if server auth enabled
- `NEUROFIELD_CHECKPOINT` — server-side weights path

## Branching

- `main` — releasable; CI must pass
- feature branches → PR → review → merge
- never commit `*.safetensors` (see `.gitignore`)

## Versioning

- Package version in `pyproject.toml`
- Tag releases `vMAJOR.MINOR.PATCH`
- Breaking API / checkpoint format → MAJOR
