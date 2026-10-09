# AERIS / NeuroField — Full component guide

**Purpose:** Har component kya hai, kaise kaam karta hai, kis order mein chalta hai.  
**Audience:** Tum (owner) — copy / reference ke liye.  
**Status:** Architecture working; yeh operational description hai.

Contact: giriisdev@gmail.com

---

## Big picture (one token step)

```text
1. Embed
2. Dendrite          (local context + delayed error)
3. Metacognition     (monitor → k, gate_bias, want_replay)
4. Neural field      (working state h)
5. Predictor         (next-embed forecast → later surprise)
6. Skill router      (top-k skills on h)
7. Neuromodulator    (gate g — kitna likhna hai)
8. Fast memory       (short segment M)
9. Slow memory       (context-window ring slots)
10. Output mix       (skills + fast read + slow read)
11. LM head          (logits → next token)
```

Har **naya token** pe yeh loop ek baar (ya field ke andar `k` micro-steps) chalti hai.

---

## Component 1 — Embedding

**File:** `model.py` (`self.embed`)

**Kaam:** Integer token id → vector size `d_model`.

**Input:** `input_ids[:, t]`  
**Output:** `x_t` shape `(batch, d_model)`

**Note:** Agar `tie_embeddings=True` to output head wahi weight share karta hai.

---

## Component 2 — Dendrite

**File:** `modules/dendrite.py`  
**Class:** `Dendrite`

**Kaam:**
- Recent tokens ka **local causal window** (`dendrite_window`, default 3)
- Previous step ka **prediction error** inject (`gamma * error`)

**Input:**
- `x_hist` — last W embeddings
- `error` — delayed error from predictor (ya zeros pehle step)

**Output:** `d_t` — dendrite summary vector `(B, d_model)`

**Kyun:** Model ko “ab kya aa raha hai” + “pehle predict galat tha” dono ek jagah milte hain.

**Simple:** Dendrite = short local sensory summary.

---

## Component 3 — Metacognition (monitor + control)

**File:** `modules/metacognition.py`  
**Class:** `Metacognition`  
**Call:** `meta.step(...)`

**Kaam — do phases:**

### Monitor
| Signal | Matlab |
|--------|--------|
| Surprise `S` | Prediction error magnitude |
| `bar_S` | Running baseline (per session) |
| `rel_S = S / bar_S` | Relative surprise |
| Router entropy | Conflict / uncertainty |
| Confidence | Feeling-of-knowing ≈ σ(−(rel_S−1)) |
| Arousal | 0.7·rel_S + 0.3·conflict |

### Control
| Output | Matlab |
|--------|--------|
| `k` | Field compute depth (1…k_max) — “zyada socho” |
| `gate_bias δ` | Memory write strength bias |
| `want_replay` | Consolidate / slow replay request |

**Kyun:** Human-like “monitor then control” — rules + light scalars, full learned critic nahi.

**Simple:** Meta decide karti hai kitna effort aur kitna yaad rakhna.

---

## Component 4 — Neural field

**File:** `modules/field.py`  
**Class:** `NeuralField`

**Kaam:** Working state `h` update — dendrite + memory read + `k` internal steps.

**Input:** `h`, `d_t`, memory read `r`, effort `k`  
**Output:** new `h` `(B, d_model)`

**Simple:** Field = abhi ka “sochne wala” recurrent state.

---

## Component 5 — Predictor

**File:** `modules/predictor.py`  
**Class:** `Predictor`

**Kaam:**
- `h` se **next embedding** forecast `x̂_{t+1}`
- Jab `x_{t+1}` aaye tab error: `e = target.detach() − pred` (default detach)
- Surprise `S = mean(|e|)` (detached)

**Timing (leakage-safe):**
- Predict at end of step `t`
- Error use starts at step `t+1`
- Same-step logits ko future token nahi dikhta

**Simple:** Predictor = “agli input ka andaza” + surprise signal.

---

## Component 6 — Skill router

**File:** `modules/router.py`  
**Classes:** `SkillOperator`, `SkillRouter`

**Kaam:**
- `n_skills` chhote MLP operators
- Softmax / top-k routing weights `π`
- Output `o_t` = weighted skill mix
- Entropy → metacognition conflict signal

**Input:** `h`  
**Output:** `o_t`, `π`, entropy

**Simple:** Skills = local compute tools; router = kaun se tools on.

---

## Component 7 — Neuromodulator (gate)

**File:** `modules/neuromodulator.py`  
**Class:** `Neuromodulator`

**Kaam:** Write gate `g ∈ (0,1)`:

```text
g = σ( a_s · (rel_S − 1) + a_c · |h| + bias + δ_meta )
```

- High surprise → stronger write  
- Meta `gate_bias` encoding priority badalti hai  
- `gate_cap` safety se clamp

**Output:** `g` shape `(B, 1)`

**Simple:** Gate = “kitna yaad mein likho”.

---

## Component 8 — Fast memory

**File:** `modules/memory.py`  
**Class:** `FastMemory`

**Kaam — short-term segment memory:**

```text
k = normalize(W_k · d_prev)
v = W_v · o_t
M ← Λ · M + g · k (v − Mᵀk)ᵀ     # delta-rule
r = Mᵀ · normalize(W_q · d_t)    # read
```

- `M` shape `(B, d_k, d_v)` **per sequence**
- `Λ = sigmoid(log_lambda)` multi-timescale decay
- Zero `d_prev` pe write skip
- Segment length ≈ `truncate_write_window` (e.g. 32)

**Simple:** Fast = abhi ke segment ki detail memory.

---

## Component 9 — Slow memory (context window)

**File:** `modules/memory.py`  
**Class:** `SlowMemory`

**Kaam — long context as ring of slots:**

```text
Every truncate_window tokens:
    slots[ptr] ← snapshot of fast M
    ptr ← (ptr + 1) % n_slots

Read:
    query all filled slots (content-addressed)
    + small global prior M_s
```

| State | Shape / role |
|-------|----------------|
| `slow_slots` | `(B, n_slots, d_k, d_v)` context window |
| `slow_ptr` | next write index |
| `slow_filled` | which slots valid |
| `n_slots` | ≈ `max_seq_len / segment` (or `context_slots`) |

`W_q` init pe fast `W_k` se tie — retrieve jo write hua.

**Legacy:** `replay()` ab bhi global `M_s` prior update kar sakta hai (trainer).

**Simple:** Slow = context window store; fast = current page.

---

## Component 10 — Output mix + head

**File:** `model.py`

```text
combined = LayerNorm( o_t + mem_scale · P_f(r_fast) + P_s(u_slow) )
logits = head(combined)
```

- `P_f` / `P_s` project memory reads → `d_model`
- `mem_scale` learnable — memory path strength
- `head` → vocab logits (tied embed optional)

**Simple:** Skills + short memory + context memory → next-token scores.

---

## Component 11 — Safety / audit (side path)

**File:** `safety/monitor.py` + model safety caps

- Write-norm / memory-norm caps (per-sequence where applied)
- Optional anomaly freeze (`freeze_writes`)
- Audit dict: gates, norms, k, surprises

**Simple:** Guardrails taaki writes explode na hon.

---

## Supporting systems (model ke bahar, product)

| System | Role |
|--------|------|
| **Tokenizer** | text ↔ ids (char / BPE; specials `<|user|>` …) |
| **Checkpoint I/O** | **only** `model.safetensors` + JSON sidecars |
| **Trainer** | next-token loss, optional slow replay |
| **RLHF** | reward / DPO / PPO modules (small scale) |
| **Server** | FastAPI chat, sessions, adaptive temp/length |
| **Sandbox** | optional tool shell/python isolation |

---

## State carried across tokens (session)

```text
h, M, error, r, d_prev, x_hist
pending_pred, bar_S, last_router_ent
slow_slots, slow_ptr, slow_filled
```

Serve pe har chat session apna state rakh sakta hai (isolation).

---

## Order cheatsheet (copy)

```text
1 Embed
2 Dendrite
3 Metacognition
4 Neural field
5 Predictor (for next step)
6 Skills + router
7 Neuromodulator gate
8 Fast memory write/read
9 Slow memory slot write (segment end) + read
10 Output + LM head
11 Safety/audit metrics
```

---

## Related docs

| Doc | Topic |
|-----|--------|
| CONTEXT_SLOW_MEMORY.md | Slow = context window detail |
| MEMORY_FIXES.md | Write-norm, decay, detach |
| TRAINING_PROCESS_COMPANY_STYLE.md | How to train |
| MULTI_MACHINE_TRAINING.md | PC + Colab weights |
| DYNAMIC_SAMPLING.md | Serve temp / max tokens |

---

*End of component guide — keep this file with the repo as the single reference for “har piece kaise kaam karta hai”.*
