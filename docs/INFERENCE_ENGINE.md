# AERIS_64 / SENTINEL — Inference Engine Architecture

DeepInfer-style map of **this** codebase: what exists today, what is partial, and what is roadmap only.

**Status (honest):** research / single-user / Colab-ready — **not** multi-tenant production parity with DeepInfer/vLLM.

---

## 1. High-level picture

```text
                    ┌─────────────────────────────┐
  HTTP clients  ──► │  FastAPI create_app          │
                    │  serving/server.py           │
                    │  routes + SamplingParams     │
                    └──────────────┬──────────────┘
                                   │ ADD → SCHEDULE
                                   ▼
                    ┌─────────────────────────────┐
                    │  Serial engine_run           │
                    │  (threading.Lock scheduler)  │
                    └──────────────┬──────────────┘
                                   │ PREFILL / DECODE
                                   ▼
                    ┌─────────────────────────────┐
                    │  Model executor              │
                    │  SENTINEL.forward + state    │
                    │  modules/* (field, memory…)  │
                    └──────────────┬──────────────┘
                                   │ logits
                                   ▼
                    ┌─────────────────────────────┐
                    │  Sampler                     │
                    │  temp / top_p / top_k / seed │
                    └──────────────┬──────────────┘
                                   │ FINALIZE
                                   ▼
                         { text, finish_reason, latency_ms }
```

| Layer | Path |
|-------|------|
| API | `packages/sentinel/serving/server.py` |
| Lifecycle | `add_request` → `engine_run` |
| Model | `packages/sentinel/model.py` (`SENTINEL`) |
| Modules | `packages/sentinel/modules/` |
| Checkpoint I/O | `packages/sentinel/checkpoint/` |
| Train parallel (not serve) | `packages/sentinel/training/parallelism.py` |

---

## 2. Main pieces

| Piece | Role | Maturity |
|-------|------|----------|
| **API front-end** | `/v1/chat`, `/v1/chat/completions`, `/v1/completions` | OK |
| **Sequence** | One job: ids, state, status, output | OK |
| **SamplingParams** | max_tokens, temperature, top_p, top_k, seed | OK |
| **Scheduler** | Global `threading.Lock` — one RUNNING seq | Basic |
| **Prefill** | `forward(prompt_ids, state=None)` | OK |
| **Decode** | `forward([token], state)` loop | OK |
| **Sampler** | greedy / multinomial + top_p/k | OK |
| **Tokenizer** | from checkpoint folder | OK |
| **Model executor** | SENTINEL + modules | OK |
| **Paged KV blocks** | N/A — recurrent `state` dict | Different design |
| **Multi-worker GPU** | Not in serve | Roadmap |
| **Continuous batching** | No | Roadmap |
| **Structured output** | Partial (stop markers only) | Roadmap |
| **MTP / speculative** | No | Roadmap |

### Request lifecycle

```text
1. ADD       — normalize prompt, encode, Sequence(WAITING)
2. SCHEDULE  — acquire lock, Sequence(RUNNING)
3. PREFILL   — full prompt forward, sample token₁
4. DECODE    — sample/append loop until stop or max_tokens
5. FINALIZE  — decode text, cut markers, FINISHED + finish_reason
```

`finish_reason`: `stop` | `length` | `error`.

---

## 3. Model executor

Not a separate CUDA graph runner — **one PyTorch module**:

```text
input_ids (B, T)
  → Embedding
  → Dendrite local windows
  → for j in 1..k_depth:
        hierarchical / gate signals
        SkillRouter (top-k)
        FastMemory write
        NeuralField step
  → SlowMemory ring interaction
  → logits (B, T, V)
  → ModelOutput(logits, state=…)
```

| Module | Executor role |
|--------|----------------|
| `dendrite` | short temporal context |
| `memory` | FastMemory + SlowMemory ring |
| `field` | recurrent field depth |
| `router` | sparse skills |
| `predictor` | multi-timescale heads (train-heavy) |
| `neuromodulator` | gates |
| `metacognition` | optional control |
| `workspace` | J-space / lens (optional learned) |
| `inference_cache` | optional cache helpers |

**KV vs state:** DeepInfer pages Transformer KV. AERIS carries a **fixed-size state** (fast + slow + pass tensors). Long context does not grow a linear KV tensor the same way.

Train and decode share the **same `forward()`** (`T>1` prefill / train, `T=1` decode).

---

## 4. Parallelism

| Mode | Training | Inference (serve) |
|------|----------|-------------------|
| Single device | Yes | **Yes (default)** |
| DDP data parallel | Yes (`torchrun`) | No |
| Tensor parallel | Partial mesh hooks | No |
| Pipeline parallel | Partial mesh hooks | No |
| MoE expert parallel | N/A (small skill router) | N/A |
| Attention CP / DP | No | No |

Inference today: **one process, one device, one lock**.  
For production multi-GPU serve you would add worker processes + sharded weights; not implemented.

---

## 5. Speculative / MTP / DSpark

| Feature | DeepInfer | AERIS |
|---------|-----------|-------|
| Multi-token prediction draft | Yes | **No** |
| Confidence-scheduled verify | Yes | **No** |
| Hierarchical predictor | N/A | Train-time multi-scale errors; **not** used as MTP draft engine |

**Roadmap (optional):** use coarse predictor or small draft head to propose 2–4 tokens, verify with full forward — only worth it when model is large enough that decode dominates.

---

## 6. Sampling & structured output

### Sampling (implemented)

```text
temperature ≤ ε  →  argmax
else:
  logits /= temperature
  top_k filter (optional)
  top_p nucleus (optional)
  multinomial
  optional Generator(seed)
```

Stop:

- special token ids when tokenizer maps markers to one id  
- text cut on `\nUser:`, `<|user|>`, …

### Structured output (not production)

| Goal | Status |
|------|--------|
| Regex / JSON schema / EBNF masks | **Not implemented** |
| Stop at role markers | Yes |
| “JSON mode” constrained decode | Roadmap |

DeepInfer-style structured output needs a **token mask each step**. AERIS can add later via `so_args` on `SamplingParams` without changing `forward`.

---

## 7. Async scheduling

| DeepInfer | AERIS |
|-----------|-------|
| CPU schedule overlapped with GPU forward | FastAPI async routes, **sync** `engine_run` under lock |
| Zero-overhead batch scheduler | Serial only |

Current behavior:

```text
Request A:  [========== generate ==========]
Request B:            wait(lock)  [==== generate ====]
```

**Roadmap:**

1. `run_in_threadpool(engine_run)` so the event loop is not blocked  
2. Optional queue with max concurrency = 1 or 2  
3. True continuous batching only if batching different sequences in one `forward` (large change)

---

## 8. Production readiness

| Criterion | Ready? |
|-----------|--------|
| Single-user local / Colab demo | **Yes** |
| Checkpoint load (`model.safetensors`) | Yes |
| Health / ready probes | Yes |
| Model-only (no tool intercept) | Yes |
| Auth / rate limit / TLS | Partial / env legacy only — treat as **no** for public |
| Multi-tenant isolation | **No** |
| Horizontal scale | **No** |
| Batching throughput | **No** |
| Output quality | **Depends on weights + SFT**, not engine |
| Observability (metrics, traces) | Minimal (`latency_ms`) |

**Verdict:** production-ready as a **research inference service for one trusted user**.  
**Not** production-ready as a public or multi-tenant LLM platform.

Minimum before calling it “prod small internal”:

1. Strong SFT checkpoint (coherent answers)  
2. Auth on API  
3. Threadpool + timeouts  
4. Logging / metrics  
5. Load test single GPU  

---

## 9. API quick reference

```bash
export NEUROFIELD_CHECKPOINT=runs/colab/final
export PYTHONPATH=packages
uvicorn sentinel.api.app:create_app --factory --host 0.0.0.0 --port 8000
```

```bash
curl -s localhost:8000/
curl -s localhost:8000/readyz
curl -s localhost:8000/v1/chat -H 'Content-Type: application/json' \
  -d '{"prompt":"hello","max_new_tokens":64,"temperature":0.7,"top_p":0.9}'
```

OpenAI-style: `POST /v1/chat/completions` with `messages`.

---

## 10. Comparison slide

| Topic | DeepInfer | AERIS now |
|-------|-----------|-----------|
| High-level | Engine + workers + KV | FastAPI + SENTINEL state |
| Main pieces | Scheduler, BlockManager, Worker | Sequence, Lock, engine_run |
| Model executor | Custom kernels / MoE | PyTorch SENTINEL modules |
| Parallelism | TP / DP / EP | Train-only DDP hooks |
| Speculative | MTP / DSpark | None |
| Sampling | Full + structured | temp/top_p/k/seed + stops |
| Async schedule | Yes | Serial lock |
| Production | Large-scale internal | Research single-stream |

---

*See also: `docs/TRAINING_PIPELINE.md` for pretrain → SFT → preference.*
