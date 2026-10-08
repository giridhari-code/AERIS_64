# Production patch — sandbox + contribution ablation

Apply these files onto your AERIS_64 / NeuroField tree. **No demo / Tiny re-implementation.**

## What changed

### 1. `src/neurofield/config.py`
- Added `AblationConfig` (defaults = production: all modules on)
- Added `truncate_write_window` on `SafetyConfig` (was only via getattr before)
- `NeuroFieldConfig` carries optional `ablation` block

### 2. `src/neurofield/model.py`
- `NeuroField(..., ablation: AblationConfig | None = None)`
- `set_ablation(cfg)` for hot-swap without rebuilding weights
- Forward path respects:
  - `use_fast_memory` — skip write/read, zero fast vector
  - `use_slow_memory` — zero slow read
  - `use_neuromodulator` — fixed `fixed_gate` instead of neuromod
  - `use_metacognition` — fixed `fixed_k` instead of surprise schedule
- Default ablation = full production behaviour (no behaviour change if unused)

### 3. `src/neurofield/sandbox/` (new)
- `executor.py` — subprocess jail: CPU/RAM/timeout, bin allow-list, no network, audit
- `session.py` — per-session registry, call budget, TTL eviction
- Package path: `from neurofield.sandbox import SandboxRegistry`

### 4. `src/neurofield/serving/server.py`
- Routes (when `NEUROFIELD_SANDBOX` ≠ `0`):
  - `POST /v1/tools/shell`
  - `POST /v1/tools/python`
  - `GET  /v1/tools/audit`
  - `POST /v1/tools/reset`
- `/v1/reset` also drops the session sandbox
- Env:
  - `NEUROFIELD_SANDBOX=0` to disable
  - `NEUROFIELD_SANDBOX_TIMEOUT`, `_NETWORK`, `_MAX_SESSIONS`, `_MAX_CALLS`

### 5. `scripts/contribution_ablation.py`
- Imports **real** `neurofield.model.NeuroField`
- Section 7 recall task, multi-seed, JSON report

### 6. `tests/test_ablation_and_sandbox.py`
- Grad path, no-fast write norms, sandbox allow/deny, session budget

## How to merge

```bash
# from your repo root
cp -r path/to/AERIS_production_patch/src/neurofield/sandbox src/neurofield/
# then copy patched files:
#   config.py, model.py, serving/server.py
#   scripts/contribution_ablation.py
#   tests/test_ablation_and_sandbox.py
```

Or diff against your tree and apply by hand.

## Run contribution check (real model)

```bash
PYTHONPATH=src python scripts/contribution_ablation.py \
  --steps 800 --seeds 0,1,2 --batch 16 --d-model 64 \
  --ablations full,no_fast_mem,no_slow_mem,no_neuromod,no_metacog,skills_only \
  --device cuda --out runs/contribution.json
```

Paper reference needed **800 steps** for 1.000 recall. Short CPU runs (<400 steps) are not decisive.

## Verified in this environment

- Ablation hooks: full forward+grad OK; `no_fast` → write_norms all 0; fixed_gate=0 → gates all 0
- Sandbox: `echo production` OK; allow-list denies `curl`
- 250-step CPU probe (not conclusive): full 0.434 / no_fast 0.445 / skills_only 0.434
