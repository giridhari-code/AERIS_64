# Scalability · Maintainability · Operability

How to grow AERIS / NeuroField without turning the repo into chaos.

---

## 1. Goals

| Pillar | Meaning |
|--------|---------|
| **Scalable** | More users, longer context, larger models — without rewrite from zero |
| **Maintainable** | New contributors can change one module without breaking all |
| **Reliable** | Predictable deploys, rollbacks, eval gates |
| **Observable** | Know when chat/tools/train fail |

---

## 2. Architecture scale levers

| Layer | Scale by | Do not |
|-------|----------|--------|
| **Model size** | `preset` tiny→small→medium→large; more data first | Jump to 1B with empty data |
| **Context** | `max_seq_len`, slow `n_slots`, segment length | Expect dendrite W=3 to hold 10k tokens |
| **Traffic** | Multiple uvicorn workers **or** separate API nodes + one GPU worker | One process + blocking torch on async without queue |
| **Tools** | Cache weather/search; timeouts; circuit break | Unbounded parallel external calls |
| **Data** | `data/ready/`, versioned sets, legal only | Silent overwrite of training corpus |

### Horizontal serve (sketch)

```text
[Load balancer]
    → API replica 1 (CPU chat + tools)
    → API replica 2
         ↓
    shared CHECKPOINT volume (read-only safetensors)
```

Env per replica:

```bash
NEUROFIELD_CHECKPOINT=/models/AERIS_main
NEUROFIELD_WEB_TOOLS=1
NEUROFIELD_API_KEYS=key1,key2
WEB_CONCURRENCY=2   # gunicorn/uvicorn workers — careful with GPU
```

**GPU:** usually **one process owns the GPU**; queue inference requests rather than N workers each loading the model.

### Vertical train

```text
grad-accum ↑  before batch size ↑
bf16 / fp16 on capable GPUs
preset ladder + cosine + warmup (see SCALE_AND_OPTIMIZE.md)
```

---

## 3. Maintainability rules

1. **One official checkpoint** — `docs/AERIS_main` or `checkpoints/AERIS_main` (see MULTI_MACHINE_TRAINING.md).
2. **Weights not in git** — `*.safetensors` / `*.pt` gitignored; share via Drive/HF.
3. **Modules stay thin** — change `modules/memory.py` without rewriting server.
4. **Config over hardcode** — presets, env vars (`CHECKPOINT`, `NEUROFIELD_*`).
5. **Eval before promote** — E2E + red-team; fail ⇒ do not replace main.
6. **Docs next to features** — every major feature has a `docs/guides/*.md`.
7. **API versioning mindset** — prefer additive `/v1/...` fields; don’t break clients silently.

### Folder responsibilities

| Path | Own |
|------|-----|
| `src/neurofield/modules/` | Model internals |
| `src/neurofield/serving/` | HTTP API |
| `src/neurofield/tools/` | External facts |
| `scripts/train/` | Training entrypoints |
| `scripts/eval/` | E2E, red-team, metrics |
| `data/` | Datasets (legal) |
| `docs/guides/` | How-to |
| `deploy/` | Docker / k8s / systemd |

---

## 4. Reliability checklist

- [ ] `GET /readyz` (or health) before traffic
- [ ] API keys required in production
- [ ] Timeouts on web/weather/wiki tools
- [ ] `NEUROFIELD_WEB_TOOLS=0` if offline deploy
- [ ] Checkpoint + tokenizer vocab match
- [ ] Rollback = previous `runs/` folder promoted back to main
- [ ] Logs: train `tee`, serve stdout, tool errors

---

## 5. Observability (minimal)

| Signal | Where |
|--------|--------|
| Request count / errors | Server metrics counters |
| Latency | `ChatResponse.latency_ms` |
| Tool failures | Tool report strings + logs |
| Train loss | step logs / `meta.json` |
| Safety | `scripts/eval/red_team.py` |

Later: Prometheus/OpenTelemetry — not required on day one.

---

## 6. SDLC gates (maintainable releases)

```text
code → tests → train → E2E → red-team → promote main → serve
```

```bash
PYTHONPATH=src pytest tests/ -q

PYTHONPATH=src python scripts/eval/run_e2e_tasks.py \
  --checkpoint docs/AERIS_main --tasks data/e2e_tasks/full_tasks.jsonl

PYTHONPATH=src python scripts/eval/red_team.py \
  --checkpoint docs/AERIS_main
```

Only then copy run → official main.

---

## 7. Security & compliance (scale with users)

- Legal data only (`DATA_POLICY.md`)
- No pickle `.pt` load path
- Sandbox tools allow-listed
- Don’t log raw API keys / user PII
- Rate limit when exposing public IP

---

## 8. What “more” still means (backlog)

| Item | Why |
|------|-----|
| Inference queue (Redis/RQ) | Multi-user without multi-GPU load |
| Structured tool-calling JSON | Stronger than regex intent |
| Full MCP stdio server | IDE/agent clients |
| Model sharding / quant | Larger presets on one GPU |
| Canary deploys | 5% traffic to new main |

---

## 9. Related docs

| Doc | Topic |
|-----|--------|
| [SCALE_AND_OPTIMIZE.md](SCALE_AND_OPTIMIZE.md) | Train presets, cosine, accum |
| [MULTI_MACHINE_TRAINING.md](MULTI_MACHINE_TRAINING.md) | PC + Colab weights |
| [SDLC.md](SDLC.md) | Lifecycle gates |
| [WEB_TOOLS_AND_MCP.md](WEB_TOOLS_AND_MCP.md) | Live tools |
| [E2E_TASKS.md](E2E_TASKS.md) | Full path tests |
| [DISTILL_FINETUNE_REDTEAM.md](DISTILL_FINETUNE_REDTEAM.md) | Distill / SFT / safety |

---

*Scalability is a process: measure → bottleneck → change one lever → eval → promote.*
