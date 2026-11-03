# Spider-Web Looped MoE — Full Design Spec

> Personal-scale LLM (100M–1B): Transformer + MoE + Looped Depth + Dynamic Graph Routing.
> Implemented in Rust: `crates/spider-core`.

See also: [ARCHITECTURE.md](./ARCHITECTURE.md)

---

## 1. Design goals

| Goal | Target |
|------|--------|
| Parameter efficiency | ~400M total, ~150M active |
| Effective depth | 20+ (prelude + r×block + coda) |
| Stability | No divergence up to r≈9 |
| Inference | Early exit + MoE speedup |
| Expert utilization | >60% per expert |
| Train | ≤8GB VRAM (checkpoint + LoRA path later) |

Non-goals: frontier scale, multimodal, multi-node (v1).

---

## 2. Stack

**Prelude (2 unshared)** → **Recurrent Spider-MoE (shared, r loops)** → **Coda (2 unshared)** → **LM head**.

Recommended `r = 6` for personal scale.

---

## 3. One loop (Spider-MoE)

1. Router: `h W_r + b_r[t]` → TopK  
2. Edge generator: loop embed → sparse `A_t` (CSR)  
3. Sparse message dispatch over edges  
4. Expert FFNs (top-k, grouped)  
5. Epilogue: `h + α Δ + β x₀` + LN  
6. Early exit (infer): confidence gate  

---

## 4. Stability fixes

- `α` residual scale  
- `β` input reinjection  
- Per-loop router bias + loop embedding  
- Dynamic graph (anti-collapse)  

Monitor variance, norms, expert load in training.

---

## 5. Train vs infer

- **Train:** full `r`, no early exit, BPTT, aux load loss  
- **Infer:** early exit, optional continuous batching later  

---

## 6. Open questions (ablate first)

1. Dynamic graph vs fixed adjacency  
2. Optimal edges_per_expert  
3. Untied attention in loop worth cost?  
4. Early exit quality impact  
5. `r` vs convergence  
6. Edge loss vs load-balance interaction  

Start at 50–100M before scaling.

---

## 7. File map (Rust)

```text
crates/spider-core/src/
├── lib.rs
├── config.rs
├── model/
│   ├── mod.rs
│   ├── prelude.rs
│   ├── recurrent.rs
│   ├── coda.rs
│   ├── early_exit.rs
│   └── spider_moe.rs
├── graph/
│   ├── mod.rs
│   ├── edge_generator.rs
│   └── adjacency.rs
├── moe/
│   ├── mod.rs
│   ├── router.rs
│   ├── experts.rs
│   └── load_balance.rs
└── stability/mod.rs
```

---

## 8. References

Huginn (recurrent depth), LOOM / Foil (looped MoE), GraphMoE-style dynamic edges, Candle (Rust tensors).

---

*Version 1.0 · Status: scaffold + design; ablation required before scale-up*
