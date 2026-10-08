# AERIS_64 — Model Card

**Architecture:** NeuroField v2 (two-speed memory). **Product name:** AERIS_64.


## Model details

| Field | Value |
|-------|--------|
| Name | **AERIS_64** |
| Architecture | NeuroField v2 (two-speed memory) — unchanged |
| Product version | 2.2.0 |
| Tokenizer | Character (default) or BPE |
| Bundled demo checkpoints | ~36k–63k parameters (measured from tensor shapes) |
| 1B configuration | `configs/aeris_1b.yaml` — 1,004,511,364 parameters, **no trained weights shipped** |
| License | MIT (see LICENSE) |

## Intended use

- Research and demos of **two-speed memory** (fast delta-rule + slow backprop)
- Internal API prototype with auth, rate limits, streaming
- Hindlish / short Hindi phrase experiments
- Educational production packaging (Docker/K8s stubs)

## Out of scope

- **Not** a foundation model at GPT/Claude/Grok scale
- Not reliable for medical, legal, or financial advice
- Not a replacement for multilingual production LLMs
- Character-level mode has weak long-form coherence

## Training data

- `data/company_corpus.txt` — identity, API help, support phrases
- `data/india_multilang.txt` — Hindlish, Hindi samples, typos
- `data/skills_real.txt` — soft-skills text
- Optional user-provided UTF-8 corpora

## Evaluation

```bash
PYTHONPATH=src python scripts/eval_company.py --checkpoint docs/neurofield_company
```

Metrics: held-out cross-entropy / perplexity (only when `--val-data` is supplied), optional identity-string hit, sample generations.
Older `eval_report.json` files in `docs/` computed val on the training corpus and are not meaningful.

## Safety

- Optional API keys (`NEUROFIELD_API_KEYS`)
- Rate limiting (`NEUROFIELD_RATE_LIMIT`)
- Safety monitor + optional write freeze on anomaly
- No claim of full alignment or red-team completion

## Limitations (honest)

1. Small capacity → mixes topics and invents text
2. Limited Indian-language coverage without large data
3. CPU demo latency ~1–2s per short completion
4. Session memory is in-process only (lost on restart)

## Contact / ownership

NeuroField Contributors — research prototype productized for deployment practice.
