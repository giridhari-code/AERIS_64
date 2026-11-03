# Spider-Web Looped MoE — System Architecture

**Status:** research → personal-scale target (100M–1B)  
**Principle:** Capacity (MoE) × Depth (loops) × Diversity (dynamic graph)  
**Implementation:** Rust (`crates/spider-core`) — Python SENTINEL retired

Formerly: AERIS / SENTINEL (two-speed memory + neural field). That stack is **legacy** under `packages/sentinel/` and is not the active model path.

---

## 1. Vision

Compute-efficient LLM on a single consumer GPU:

| Axis | Mechanism | Effect |
|------|-----------|--------|
| **Capacity** | Mixture-of-Experts | params ↑, active compute flat |
| **Depth** | Shared-weight loops (`r`) | effective depth ↑, params flat |
| **Diversity** | Spider-web sparse expert graph | per-loop paths, anti-collapse |

---

## 2. Deployables

| # | Deployable | Path | Role |
|---|------------|------|------|
| 1 | **LM (Spider-Web)** | `crates/spider-core` | Model, train hooks, forward |
| 2 | **Artifacts** | `runs/`, checkpoints | Weights, tokenizer |
| 3 | **Serve API** | TBD (Rust or thin Python edge) | `/v1/chat`, health |
| 4 | **Frontend** | `frontend/` | UX → API |
| 5 | **0zcode** | `apps/0zcode` | Terminal agent (unchanged SKU) |

Training process stays separate from live chat.

---

## 3. High-level stack

```text
INPUT [B, T]
    │
    ▼
TOKEN EMBEDDING
    │  x₀
    ▼
╔══════════════ PRELUDE ══════════════╗
║  2 unshared Transformer layers      ║
╚════════════════╤════════════════════╝
                 │  h₀
╔════════════════▼════════════════════╗
║  RECURRENT BLOCK (shared, × r)      ║
║  for t = 1..=r:                     ║
║    router → edgegen → dispatch      ║
║    → expert FFN → epilogue          ║
║    → early exit? (infer only)       ║
╚════════════════╤════════════════════╝
                 │  h_r
╔════════════════▼════════════════════╗
║  CODA — 2 unshared Transformer      ║
╚════════════════╤════════════════════╝
                 │
                 ▼
            LM HEAD → logits
```

**Effective depth (r=6):** Prelude 2 + Recurrent ~24 + Coda 2 ≈ **28** layer-equivalents with ~8 stored layer blocks.

---

## 4. Spider-MoE layer (one loop)

1. **Router** — `logits = h W_r + b_r[t]`, TopK → expert indices + weights  
2. **Edge generator** — loop embed `e_t`, sparse adjacency `A_t` (CSR)  
3. **Sparse dispatch** — message over graph neighbors  
4. **Expert FFN** — grouped top-k experts  
5. **Epilogue** — `u = h + α·Δ + β·x₀`, LayerNorm  
6. **Early exit** (inference) — `σ(h · w_exit) > τ` → stop

Shared across loops: router W, edge gen W, experts, α/β.  
Per-loop: router bias `b_r[t]`, graph `A_t`, loop embed.

---

## 5. Stability

| Problem | Fix |
|---------|-----|
| Variance blow-up | Residual scale `α ≈ 0.5` |
| Representation drift | Input reinject `β · x₀` |
| Weight-sharing symmetry | Per-loop bias + loop embed |
| Expert collapse | Dynamic sparse graph |

Training monitors: `Var(h_t)`, `‖h_t‖`, expert load histogram.

---

## 6. Training vs inference

| | Training | Inference |
|--|----------|-----------|
| Loops | Always full `r` | Up to `r`, early exit |
| Graph | Per-loop learned | Same + optional per-token |
| Grad | BPTT through loops | None |
| Exit head | Off | On |

Loss: CE + aux load-balance + optional edge regularizer.

---

## 7. Parameter targets

| Scale | H | Experts E | r | Active (k=2) |
|-------|---|-----------|---|--------------|
| Proto | 512 | 8 | 6 | ~25M |
| Personal | 1024 | 16 | 6 | ~150M |
| 1B-class | 2048 | 32 | 8 | larger |

Default personal target: **~400M total / ~150M active**, ≤8GB train with checkpointing.

---

## 8. API spine (stable intent)

Same surface as before where possible:

| Method | Path | Role |
|--------|------|------|
| POST | `/v1/chat` | Conversation |
| GET | `/health` `/readyz` | Ops |

Breaking model semantics may still keep path names; document version in response metadata.

---

## 9. Repo map

```text
crates/spider-core/     # NEW model library (Rust)
docs/SPIDER_WEB_MOE.md  # Full design spec
docs/ARCHITECTURE.md    # This file
apps/0zcode/            # Terminal agent (unchanged)
packages/sentinel/      # LEGACY Python SENTINEL — do not extend
```

---

## 10. Explicit non-goals

- Frontier 100B+ training  
- Multimodal  
- Multi-node distributed (v1)  
- Keeping Python neural-field / two-speed memory as the main path  

---

## 11. Migration note

| Old (SENTINEL) | New |
|----------------|-----|
| `packages/sentinel` Python model | `crates/spider-core` |
| k-step neural field + fast/slow mem | Prelude + looped Spider-MoE + Coda |
| Skill routing | MoE router + spider graph |
| Metacognition J-space | Early-exit + stability monitors |

Weights and training recipes from SENTINEL are **not** compatible. Retrain from the Spider-Web configs.
