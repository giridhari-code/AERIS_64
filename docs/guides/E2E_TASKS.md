# Free END-TO-END tasks

Open-ended tasks through the **full path**: prompt → model (or HTTP API) → text out.

Not multiple-choice. Tiny checkpoints will fail many free tasks; the harness still tests the pipeline.

## Files

| Path | Role |
|------|------|
| `data/e2e_tasks/free_tasks.jsonl` | Task list (add your own) |
| `src/neurofield/tasks/e2e.py` | Loader + runner |
| `scripts/eval/run_e2e_tasks.py` | CLI |

## Task JSONL format

```json
{"id": "my_task", "category": "free", "prompt": "your open request", "expect_contains_any": ["optional", "keywords"], "max_tokens": 96, "notes": "..."}
```

- Empty `expect_contains_any` → graded as **FREE** (no auto pass/fail).
- Any match → **PASS**, else **FAIL**.

## Run local checkpoint

```bash
PYTHONPATH=src python scripts/eval/run_e2e_tasks.py \
  --checkpoint docs/AERIS_main \
  --tasks data/e2e_tasks/free_tasks.jsonl \
  --device cpu
```

## Run against running server

```bash
PYTHONPATH=src python scripts/eval/run_e2e_tasks.py \
  --api http://127.0.0.1:8000 \
  --tasks data/e2e_tasks/free_tasks.jsonl
```

## Add your free task

Edit `free_tasks.jsonl`:

```json
{"id": "shop_faq", "category": "free", "prompt": "Shop kitne baje khulti hai?", "expect_contains_any": ["10", "baje", "subah"], "max_tokens": 64, "notes": "domain FAQ"}
```

## Honest limits

Free E2E **quality** needs trained weights + domain data.  
This add = **task harness + sample free tasks**, not a new brain.
