# NeuroField / AESC v2.4 — Research Note

**Two-Speed Memory with Per-Token Dendritic Loop and Context-Window Slow Store**

| Field | Value |
|-------|--------|
| Project | AERIS / NeuroField |
| Version | v2.4 (architecture note) |
| Date | 2026-10-10 |
| Contact | giriisdev@gmail.com |
| Code | https://github.com/giridhari-code/AERIS_64 |

---

## Abstract

NeuroField is a recurrent language-model architecture that processes tokens **one step at a time**. Each step builds a local dendritic feature \(d_t\) from a short causal window of embeddings, updates a working field state \(h\), routes through skill operators, and writes to a **fast** associative matrix. Long context is not expanded inside the dendrite; it is stored in a **slow** ring of segment slots that implement an explicit **context window**. This note describes the per-token loop, the two-speed memory split, metacognitive control, and the limits of the design relative to large Transformer LMs.

---

## 1. Motivation

Standard Transformers keep context via attention over a token sequence. NeuroField instead uses:

1. A **fixed local window** (dendrite) for immediate sensory context.
2. A **fast weight matrix** \(M\) for the current segment (delta-rule).
3. A **slow slot ring** for multi-segment context (context window).

The design goal is leakage-safe prediction, session-local state, and interpretable memory stages—not parity with frontier LLM scale.

---

## 2. Per-token computational loop

For sequence length \(T\), the model runs \(T\) iterations. At step \(t\):

```text
x_t = Embed(token_t)
x_hist ← shift window; append x_t          # last W embeddings
d_t = Dendrite(x_hist, error_{t})          # local feature
meta → (k, gate_bias, want_replay)
h ← Field(h, d_t, r, k)                    # k inner steps
pending_pred ← Predictor(h)                # for x_{t+1}
o_t, π ← Skills/Router(h)
g ← Neuromodulator(rel_S, h, gate_bias)
M ← FastMemory.write(M, d_{t-1}, o_t, g)
r ← FastMemory.read(M, d_t)
if segment boundary: SlowMemory.write_slot(M)
u ← SlowMemory.read(d_t, slots)
logits_t ← Head( Norm(o_t + scale·P_f(r) + P_s(u)) )
```

**Figure 1 — Token loop (conceptual)**

```text
        word_t
           │
           ▼
        Embed → x_t ──► x_hist (W slots)
                           │
                           ▼
                       Dendrite → d_t
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
           Meta(k,δ)    Field → h    Predictor
                           │
                           ▼
                    Skills + Gate g
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
         Fast M (segment)        Slow slots (context)
              │                         │
              └──────────┬──────────────┘
                         ▼
                      logits_t
                         │
                         ▼
                   next token t+1  (same loop)
```

Important properties:

- \(d_t\) is **recomputed every step**; it is not a static bag-of-words vector.
- Field state \(h\) **carries over** across tokens.
- Dendrite never sees all \(T\) embeddings at once—only the last \(W\) (default \(W=3\)).

---

## 3. Dendrite (local input window)

\[
d_t = \tanh\Big(\sum_{j=0}^{W-1} c_j \odot x_{t-j} + b\Big) + \gamma \odot e_t
\]

- \(x_{t-j}\): recent embeddings (causal).
- \(e_t\): **delayed** prediction error from the previous step (leakage-safe).
- For \(T \in \{100,\ldots,1000\}\), the dendrite still uses only the last \(W\) vectors; older tokens must live in field/memory state.

---

## 4. Two-speed memory

### 4.1 Fast memory (segment)

Delta-rule (paper-style):

\[
k_t = \mathrm{normalize}(W_k d_{t-1}),\quad
v_t = W_v o_t
\]
\[
M_t = \Lambda M_{t-1} + g_t\, k_t (v_t - M_{t-1}^\top k_t)^\top
\]

- \(M \in \mathbb{R}^{B \times d_k \times d_v}\) per sequence.
- \(\Lambda = \mathrm{sigmoid}(\lambda)\) multi-timescale decay.
- Segment length ≈ `truncate_write_window` (e.g. 32).

### 4.2 Slow memory (context window)

**Figure 2 — Context window as slot ring**

```text
Fast M (current segment)
         │  every segment_len tokens
         ▼
   ┌───┬───┬───┬───┬───┐
   │ S0│ S1│ S2│…│S_{n-1}│   ← n_slots ≈ max_seq_len / segment_len
   └───┴───┴───┴───┴───┘
         │
         ▼  content-addressed read
        u_t  (+ optional global prior M_s)
```

- At segment boundaries, a snapshot of fast \(M\) is written into the next ring slot.
- Read queries **all filled slots**, so long context is handled on the slow path.
- Query basis is tied to fast keys at init so retrieval matches what was written.

---

## 5. Metacognition (monitor → control)

| Monitor | Control |
|---------|---------|
| Relative surprise \(S/\bar S\) | Effort \(k\) (field depth) |
| Router entropy (conflict) | Gate bias \(\delta\) (encoding strength) |
| Confidence / arousal | `want_replay` (consolidation request) |

Neuromodulatory gate:

\[
g = \sigma\big(a_s(\tilde S-1) + a_c |h| + b + \delta_{\mathrm{meta}}\big)
\]

This is a **rule-based control layer**, not a full learned inner critic.

---

## 6. Leakage safety

The predictor forecasts \(x_{t+1}\) from state after token \(t\). The error is applied only when \(x_{t+1}\) arrives. Nothing derived from \(x_{t+1}\) enters the logits that predict \(x_{t+1}\).

---

## 7. Empirical notes (prior v2.3 runs)

On a small production configuration (\(d_model=64\)), reported 4-pair recall eval:

| Condition | Acc (800 steps) |
|-----------|-----------------|
| Full model | 0.973 |
| No fast memory | 0.344 |

Ablations at 400 steps showed metacognition and neuromodulation contributing measurable deltas; slow memory contribution was small in that suite—motivating the v2.4 explicit **context-ring** design. **Long-horizon hypothesis (H1) remains under-tested.**

---

## 8. Limits (honest)

1. Fixed \(d_k \times d_v\) capacity vs Transformer KV growth.
2. Dendrite \(W\) is local only; very long context depends on slot count and eviction.
3. Skills are per-token MLPs, not full sequence attention.
4. Demo checkpoints are not general assistants; capability tracks data and train scale.

---

## 9. Implementation map

| Module | Path |
|--------|------|
| Core loop | `src/neurofield/model.py` |
| Dendrite | `src/neurofield/modules/dendrite.py` |
| Field | `src/neurofield/modules/field.py` |
| Fast/Slow memory | `src/neurofield/modules/memory.py` |
| Metacognition | `src/neurofield/modules/metacognition.py` |
| Neuromodulator | `src/neurofield/modules/neuromodulator.py` |
| Router | `src/neurofield/modules/router.py` |
| Component guide | `docs/COMPONENTS_FULL_GUIDE.md` |

Checkpoints: **`model.safetensors` only** (no pickle `.pt` load path in current I/O policy).

---

## 10. Conclusion

NeuroField v2.4 makes the **per-token loop** and **context-window slow store** explicit: dendrite stays local; long context is a ring of segment memories; field state \(h\) links steps. The architecture is coherent for research on two-speed memory and metacognitive gating. It is not a claim of frontier LLM equivalence.

---

## References (internal)

- AESC v2.3 Technical Report (`docs/paper/AESC_v2_3_Technical_Report.md`)
- Context slow memory note (`docs/CONTEXT_SLOW_MEMORY.md`)
- Training process (`docs/guides/TRAINING_PROCESS_COMPANY_STYLE.md`)

---

*This document is a research / engineering note for the AERIS personal project, not a peer-reviewed publication.*
