# Changelog

## 2.3.0 — 2026-10-09

### Repository
- Professional layout: `docs/{paper,guides,reference,audits,archive}`, `scripts/{train,eval,data,research}`, `checkpoints/examples`
- Incomplete weight-less folders moved out of `docs/` into `checkpoints/examples/`
- Clean root README and docs indexes

### Model / research
- Fast/slow memory production fixes (decay, W_q/W_k, zero slow init, gate bias)
- Brain-inspired metacognition (monitor + control)
- Tool sandbox (`/v1/tools/*`)
- Contribution ablation script
- AESC v2.3 technical report under `docs/paper/`

## 2.2.0

- Prior NeuroField / AERIS public tree (audits, 1B config recipe, serving stack)
