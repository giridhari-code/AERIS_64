# SI_v0 — Synthetic Intelligence (minimal reliable layer)

## Definition

```text
SI_v0 = tools + budgets + audit + optional NeuroField generate
```

Not AGI. Success = fixed suite passes + live facts from tools.

## Components

| Piece | Path |
|-------|------|
| Agent loop | `src/neurofield/agent/loop.py` |
| Checkpoint generate | `src/neurofield/agent/generate.py` |
| CLI | `scripts/agent/run_agent.py` |
| Suite | `scripts/eval/si_v0_suite.py` |
| HTTP | `POST /v1/agent` |

## Run

```bash
# tools only (reliable facts)
PYTHONPATH=src python scripts/agent/run_agent.py "calculate 15*3"

# + model wording (needs trained ckpt)
PYTHONPATH=src python scripts/agent/run_agent.py "hello" \
  --checkpoint docs/runs/chunk_v2 --device cpu

# suite
PYTHONPATH=src python scripts/eval/si_v0_suite.py
```

## Checklist

- [x] max_steps / tool fail budget
- [x] tool audit events
- [x] calc/time/weather/search routing (orchestrate)
- [x] CLI + `/v1/agent`
- [x] eval suite skeleton
- [ ] large SFT competence (your data)
- [ ] multi-step plan JSON (SI_v1)

## Next SI_v1

Plan → tool → verify → answer; task success %.
