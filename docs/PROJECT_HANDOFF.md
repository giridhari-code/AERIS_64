# SENTINEL (AERIS_64) — Project handoff

**Date:** 2026-10-12  
**Author:** Giridhari Karmakar · giriisdev@gmail.com  
**Repo:** https://github.com/giridhari-code/AERIS_64

## Verdict

| Area | Status |
|------|--------|
| Architecture design | **OK** — two-speed memory, dendrite, field, router, parallel/looped path |
| Core code | **Runs** — train / checkpoint / serve path present |
| Research paper | **docs/paper/AESC_NeuroField_Full_Research_Paper_Giridhari_Karmakar.pdf** (diagrams + formulas) |
| Production frontier LM | **Not claimed** — needs data, compute, eval |

## Canonical artifacts

- Code: `src/neurofield/` (model parallel path)
- Train: `scripts/train/train_company.py`
- Paper: `docs/paper/AESC_NeuroField_Full_Research_Paper_Giridhari_Karmakar.pdf`
- Figures: `docs/paper/figures/`
- Audit: `docs/audits/FINAL_PROJECT_AUDIT_2026-10-12.md`

## Do not treat as quality proof

- Empty or 1-token train runs
- Garbage chat from tiny/repeated data
- Example checkpoint configs without trained weights in git

## Future work (when time allows)

1. Real pretrain + SFT data  
2. GPU throughput (batch, k, DDP)  
3. Gate stability  
4. Multi-seed eval vs Transformer baseline  
5. Stronger safety policy  

## Architecture one-liner (SENTINEL)

Embed → causal dendrite → k depth (predict/gate/router/fast-M scan/field) → slow ring → logits.

**Handoff complete.** Design stands; scale is open research.


## Paper alignment

See [docs/audits/PAPER_CODE_ALIGNMENT_100.md](audits/PAPER_CODE_ALIGNMENT_100.md) and `scripts/eval/paper_alignment_smoke.py`.
