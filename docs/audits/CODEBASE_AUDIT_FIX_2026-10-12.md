# Codebase audit + fix — 2026-10-12

## Smoke

- SENTINEL parallel forward + loss + backward: OK
- Core tests: **53 passed** (delta rule, safety freeze, parallel main, decode parity, paper checklist, model, components)
- `test_company` / `test_server`: require FastAPI; now `importorskip` so collection does not hard-fail without deps

## Fixes applied

1. **model.forward signature** — clean `label_smoothing` parameter formatting  
2. **tests/test_company.py** — `pytest.importorskip("fastapi")` before FastAPI import  
3. **train_company.py** — refuse train if corpus has fewer than 8 tokens (prevents useless 1-token runs)  
4. **version** — `2.5.1`

## Architecture status

Main path remains parallel/looped SENTINEL (not Transformer). Modules: embed, dendrite, field, predictor, neuromod, router, fast_mem, slow_mem, readout.

## Known non-blocking gaps

- Full pytest suite can OOM on tiny hosts if all tests run at once  
- FastAPI must be installed for serve/API tests  
- Language quality still data-bound  

## Not broken

- Decode parity tests present and passing in core set  
- Safety freeze / Change-5 delta rule tests passing  
