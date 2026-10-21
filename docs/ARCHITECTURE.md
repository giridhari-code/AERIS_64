# AERIS / SENTINEL — System Architecture

**Status:** production-oriented target  
**Principle:** independent deployables, one API spine, tool trust tiers, offline training plane.

## 1. Deployables (5)

| # | Deployable | Path / process | Responsibility |
|---|------------|----------------|----------------|
| 1 | **LM artifacts** | `runs/`, checkpoints | Weights, tokenizer, train exports |
| 2 | **SENTINEL** | `packages/sentinel` (lib) | Model architecture, train/eval code |
| 3 | **Backend** | `uvicorn sentinel.api.app:create_app` | Auth, chat, tools, memory, health |
| 4 | **Frontend** | `frontend/` | UX only → Backend HTTP |
| 5 | **0zcode** | `apps/0zcode` (Rust) | Non-coding terminal agent |

Training never shares a process with live chat.

## 2. Runtime topology

```text
Clients: Web UI | 0zcode | SDK
            │
            ▼
     Backend API  (/v1/chat, /v1/tools, /v1/tools/call)
            │
     ┌──────┴──────┐
     ▼             ▼
 Inference      Tool runtime
 SENTINEL+LM    dispatch + tiers T0–T3
```

## 3. API spine (stable)

| Method | Path | Role |
|--------|------|------|
| POST | `/v1/chat` | Conversation |
| GET | `/v1/tools` | Catalog + dispatch names |
| POST | `/v1/tools/call` | `{ "name", "arguments" }` |
| GET | `/health` `/readyz` | Ops |

Breaking changes require `/v2`.

## 4. Tool tiers

| Tier | Meaning | Default |
|------|---------|---------|
| T0 | Public facts (weather, search, calc, …) | on |
| T1 | Read-only (git status, fetch, RAG, …) | on |
| T2 | Write / workspace (ajeebwork, kalofacts, …) | on (`NEUROFIELD_TOOLS_T2=1`) |
| T3 | Exec (bash, code_execution) | **off** (`NEUROFIELD_TOOLS_T3=1` to enable) |

Override: `NEUROFIELD_TOOLS_MAX_TIER=0|1|2|3`.

Implementation: `packages/sentinel/tools/tiers.py` + `dispatch.run_tool`.

## 5. Checkpoint promotion

```text
train → eval smoke → tag artifact → set NEUROFIELD_CHECKPOINT → restart Backend
```

Do not point prod at an un-eval’d run directory.

## 6. 0zcode

- Separate SKU; **no source-code generation**.
- Does not require Backend or LM.
- Optional future: call Backend `/v1/tools/call` for shared tools.

## 7. Explicit non-goals

- In-process training + serving
- Frontend loading weight files
- Unrestricted shell in default prod
- Dual Python+Rust 0z agents (Rust only)
