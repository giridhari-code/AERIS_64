# Full END-TO-END tasks

Complete pipeline checks: **prompt → model (or HTTP API) → text out**.

Name was briefly "free" (typo); the intended meaning is **full** E2E — whole system path, not a single-module unit test.

## Files

| Path | Role |
|------|------|
| `data/e2e_tasks/full_tasks.jsonl` | Full E2E task list |
| `src/neurofield/tasks/e2e.py` | Loader + runner |
| `scripts/eval/run_e2e_tasks.py` | CLI |

## Task format

```json
{"id": "my_task", "category": "e2e_custom", "prompt": "open request", "expect_contains_any": ["keyword"], "max_tokens": 96, "notes": "..."}
```

- Empty `expect_contains_any` → no auto grade (manual full task).
- Keyword hit → PASS, else FAIL.

## Run

```bash
# local checkpoint — full path through weights
PYTHONPATH=src python scripts/eval/run_e2e_tasks.py \
  --checkpoint docs/AERIS_main \
  --tasks data/e2e_tasks/full_tasks.jsonl \
  --device cpu

# full path through running server
PYTHONPATH=src python scripts/eval/run_e2e_tasks.py \
  --api http://127.0.0.1:8000 \
  --tasks data/e2e_tasks/full_tasks.jsonl
```

## Categories

| category | Meaning |
|----------|---------|
| `e2e_chat` | Greeting / identity through full stack |
| `e2e_arch` | Architecture recall prompts |
| `e2e_session` | Multi-step / session style |
| `e2e_custom` | Your full domain task |

## Honest note

Full E2E **harness** ≠ full intelligence. Quality still needs trained `model.safetensors` + data.
