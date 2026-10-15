# Bhutu — Metacognition manager (transcript)

**Name:** Bhutu  
**Role:** Manager inside the **Metacognition** control layer  
**Package:** `sentinel.modules.bhutu`  
**Wired in:** `/v1/chat` precheck, `SI_v1` agent plan/filter/step budget  

This document is the project transcript of what Bhutu is, what it controls, and what it does **not** do.

---

## 1. One-line definition

Bhutu is the **tool and agent traffic manager** under Metacognition.  
It does not generate user-facing prose by itself. It decides **allow / refuse / which tools / how many agent steps / RAG / sandbox**.

```text
Metacognition (neural control: k, gate_bias, replay)
    └── Bhutu (manager: tools, agent, tool_search, policy route)
            ├── Megha policy (illegal cyber → refuse)
            ├── SI_v1 plan filter + max_steps
            └── Chat gate before tools/model
```

---

## 2. Monitor → Control (same style as Meta)

| Phase | Bhutu |
|-------|--------|
| **MONITOR** | Prompt heuristics + optional confidence/arousal/conflict; Megha `evaluate_user_text` |
| **CONTROL** | `refuse` \| `route_megha` \| `no_tool` \| `use_tools` |

### Decision outputs (`BhutuDecision`)

| Field | Meaning |
|-------|---------|
| `allow` | If false, stop the turn |
| `action` | `refuse` / `route_megha` / `no_tool` / `use_tools` |
| `allowed_tools` | Only these tool names may run |
| `max_agent_steps` | Cap for SI_v1 / agent loop |
| `use_rag` | Whether retrieval is encouraged |
| `use_tool_search` | Whether tool_search is in scope |
| `allow_sandbox` | bash/code only if true (and env sandbox on) |
| `route_megha_mode` | e.g. checklist, code_review |
| `user_message` | Text shown on refuse |

Helpers:

- `filter_plan(steps)` — drop unauthorized SI_v1 steps  
- `may_run_tool(name)` — per-tool gate at execution time  

---

## 3. What Bhutu controls

| System piece | Control |
|--------------|---------|
| Illegal / unauthorized cyber chat | **Refuse** (via Megha policy) |
| Defensive security questions | **Route Megha** |
| calculator / time / weather / search / wiki / currency | Allow only if intent matches |
| SI_v1 agent | Filtered plan + step budget |
| tool_search | Only when decision says so |
| RAG | Flag `use_rag` |
| sandbox / bash / code_execution | Off unless intent + `enable_sandbox` |

---

## 4. What Bhutu does **not** control alone

| Piece | Notes |
|-------|--------|
| Neural Field **k**, neuromod **δ**, slow **replay** | Still core `Metacognition` module on the model forward |
| Model token quality | Training / data, not Bhutu |
| Offline attacks outside this API | Cannot control |

---

## 5. Runtime flow (chat)

```text
User message
    → normalize_prompt
    → bhutu_decide(text)
         refuse     → return policy message (model=bhutu-policy)
         route_megha→ Megha playbooks
         else       → SI_v1 / tools / model under allowed_tools + max steps
```

Agent path (`run_si_v1`):

1. `bhutu_decide`  
2. Refuse or Megha short-circuit if needed  
3. `plan = bhutu.filter_plan(build_plan(text))`  
4. `max_steps = min(config, bhutu.max_agent_steps)`  
5. Each tool: `may_run_tool` or skip as denied  

---

## 6. Examples (transcript)

| User | Bhutu |
|------|--------|
| `calculate 12 * 8` | `use_tools`, allowed: calculator |
| `what time is it` | `use_tools`, allowed: time |
| `how to hack someone's Instagram` | `refuse` |
| `OWASP hardening for my API` | `route_megha` (checklist / defensive) |
| casual chit-chat | `no_tool` → compose / model |

---

## 7. Code map

| Path | Role |
|------|------|
| `src/sentinel/modules/bhutu.py` | Manager implementation |
| `src/sentinel/modules/__init__.py` | Exports `Bhutu`, `bhutu_decide`, … |
| `src/sentinel/agent/si_v1.py` | Plan filter + step cap + per-tool gate |
| `src/sentinel/serving/server.py` | Chat precheck |
| `tests/test_bhutu.py` | Refuse / calc / filter / SI_v1 respect |

API entry:

```python
from sentinel.modules.bhutu import bhutu_decide, get_bhutu

decision = bhutu_decide("what time is it")
# decision.to_dict()
```

---

## 8. Design rules (production)

1. **Fail closed on policy** — illegal cyber never reaches tools.  
2. **Sandbox default cautious** — only when env allows and intent matches.  
3. **No attack tooling** — Bhutu is a manager, not a pentest engine.  
4. **Heuristics are explicit** — keyword/intent rules are documented control, not hidden demo stubs.  
5. **Meta neural path stays separate** — Bhutu manages tools/agent; `Metacognition` still owns k / δ / replay on the field.

---

## 9. Short transcript summary (copy)

> Bhutu Metacognition ke andar manager hai. Har user turn pe monitor karta hai (confidence, policy), phir control: refuse, Megha route, no tool, ya allowed tools + agent step budget. SI_v1 aur chat iske bina free tool run nahi karte. Illegal cyber block; defensive security Megha; normal facts tools allow-list se. Model ke andar field depth aur memory gate ab bhi core Metacognition module se aate hain.

---

*SENTINEL / AERIS_64 · Bhutu under Metacognition · defensive and authorized use only*
