# AERIS_64 / SENTINEL — Inference Engine Architecture

DeepInfer-style map of **this** codebase (honest: what exists vs roadmap).

**HEAD capabilities:** model-only FastAPI + `model_executor` + serial lifecycle + session-isolated J-space + production auth/rate limits.  
**Not** DeepInfer parity (no continuous batching, MTP, EP, paged KV).

---

## 1. High-level picture

```text
 HTTP / 0zcode
      │
      ▼
┌─────────────────────────────────────┐
│  FastAPI  serving/server.py         │
│  auth · rate limit · max in-flight  │
│  Sequence lifecycle                 │
└──────────────────┬──────────────────┘
                   │ ADD → SCHEDULE
                   ▼
┌─────────────────────────────────────┐
│  engine_run  (threading.Lock)       │
│  SamplingParams + stop markers      │
└──────────────────┬──────────────────┘
                   │ generate()
                   ▼
┌─────────────────────────────────────┐
│  model_executor                     │
│  loader · ModelRunner · sampling    │
└──────────────────┬──────────────────┘
                   │ prefill / decode
                   ▼
┌─────────────────────────────────────┐
│  SENTINEL.forward + session state   │
│  (fast/slow mem, field, J-space)    │
└─────────────────────────────────────┘
```

| Layer | Path |
|-------|------|
| API / lifecycle | `packages/sentinel/serving/server.py` |
| Executor | `packages/sentinel/model_executor/` |
| Model | `packages/sentinel/model.py` |
| Modules | `packages/sentinel/modules/` |
| Real-user serve | `docs/SERVING_REAL_USERS.md` |
| Train parallel | `packages/sentinel/training/parallelism.py` (not serve) |

---

## 2. Main pieces

| Piece | Status | Notes |
|-------|--------|--------|
| Sequence + SamplingParams | **OK** | `WAITING → RUNNING → FINISHED` |
| Serial scheduler | **OK** | one lock; max in-flight waiters |
| Prefill / decode | **OK** | `ModelRunner.generate` |
| Sampler | **OK** | temp, top_p, top_k, seed |
| Model executor package | **OK** | loader, runner, sampling, output |
| Session state / J-space | **OK** | rule-mode isolated (no cross-user leak) |
| Production profile | **OK** | keys, rate limit, no demo ckpt |
| Continuous batching | **No** | roadmap |
| Paged KV | **N/A** | recurrent state instead |

---

## 3. Request lifecycle (DeepInfer-shaped)

```text
1. ADD       encode prompt → Sequence(WAITING)
2. SCHEDULE  acquire lock → RUNNING
3. PREFILL   runner.prefill(prompt_ids)
4. DECODE    sample → decode(token) loop until stop / max_tokens
5. FINALIZE  detokenize, cut markers → FINISHED + finish_reason
```

Implementation: `add_request` → `engine_run` → `ModelRunner.generate`.

---

## 4. Model executor (`packages/sentinel/model_executor/`)

| Module | Role |
|--------|------|
| `model_loader.py` | checkpoint → `LoadedModel` |
| `model_runner.py` | `prefill` / `decode` / **`generate`** (tight loop, 1-token buffer) |
| `sampling.py` | `SamplingParams`, `sample_from_logits` |
| `output.py` | `ModelStepOutput` |

Serve does **not** call `model(...)` inline; it goes through the runner.

---

## 5. Parallelism

| Mode | Train | Infer (serve) |
|------|-------|----------------|
| Single device | Yes | **Yes** |
| DDP | Yes | No |
| TP / PP mesh hooks | Partial | No |
| MoE EP | N/A | N/A |

Inference = **one process, one device, serial generate**. Scale with multiple replicas + LB, not multi-worker one GPU.

---

## 6. Speculative / MTP / DSpark

| Feature | Status |
|---------|--------|
| Multi-token draft | **Not implemented** |
| Confidence-scheduled verify | **Not implemented** |
| Hierarchical predictor | Train losses / signals — **not** used as MTP engine |

Roadmap only for larger models where decode dominates.

---

## 7. Sampling & structured output

**Sampling (implemented):** temperature, top_p, top_k, seed, stop token ids, text cut on `\nUser:` / markers.

**Structured output:** stop markers only — **no** JSON-schema / regex / EBNF token masks (DeepInfer `so_args` style = roadmap).

---

## 8. Async scheduling

| DeepInfer | AERIS |
|-----------|-------|
| CPU schedule ∥ GPU | FastAPI async routes |
| Zero-overhead batcher | **Sync** `engine_run` under lock |

`NEUROFIELD_MAX_INFLIGHT` limits concurrent attempts; still one active generate. True async overlap / continuous batching = roadmap.

---

## 9. Isolation & production

- Rule-mode **J-space** lives in `Sequence.state`, not on shared module (leak fixed).
- Production: `NEUROFIELD_ENV=production` → API keys, rate limit, security headers, no demo checkpoint.
- 0zcode: `POST /v1/chat`; stream falls back if no SSE.

---

## 10. Status scorecard

| Topic | Verdict |
|-------|---------|
| High-level picture | Documented + matches code |
| Main pieces | Wired |
| Request lifecycle | DeepInfer-shaped, single-stream |
| Model executor | Package + tests |
| Parallelism (serve) | Single-device only |
| Speculative MTP | Absent (documented) |
| Sampling | Good enough |
| Structured output | Minimal |
| Async scheduling | Serial (documented) |
| Real users | Auth/limits yes; quality = checkpoint |

---

*Related: `docs/TRAINING_PIPELINE.md`, `docs/SERVING_REAL_USERS.md`.*
