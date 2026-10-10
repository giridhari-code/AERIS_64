# Distillation · Fine-tuning · Red-teaming

Three production-adjacent stages beyond plain pretrain.

---

## 1. Fine-tuning (SFT)

Continue from a checkpoint on domain / assistant data.

```bash
PYTHONPATH=src python scripts/train/train_finetune.py \
  --resume docs/AERIS_main \
  --data data/align/sft_demos.txt data/samples/english_day1.txt \
  --steps 800 \
  --preset small \
  --lr 5e-4 \
  --out docs/runs/finetune_sft \
  --device cpu
```

Also: `scripts/train/train_align.py` for preference-style align.

---

## 2. Knowledge distillation

Student learns from **teacher** soft labels (KL) + hard CE.

```bash
PYTHONPATH=src python scripts/train/train_distill.py \
  --teacher docs/AERIS_main \
  --data data/samples/english_day1.txt \
  --steps 500 \
  --temperature 2.0 \
  --alpha 0.7 \
  --out docs/runs/student_distill \
  --device cpu
```

Same vocab / head shape required for soft mode.  
For cross-architecture, generate teacher text offline then `train_finetune` on that corpus.

---

## 3. Red-teaming (safety probes)

```bash
PYTHONPATH=src python scripts/eval/red_team.py \
  --checkpoint docs/AERIS_main \
  --tasks data/redteam/prompts.jsonl \
  --device cpu
```

Or against API:

```bash
PYTHONPATH=src python scripts/eval/red_team.py --api http://127.0.0.1:8000
```

Add probes in `data/redteam/prompts.jsonl`.  
`expect_refuse: true` → output should refuse harmful asks.

**Note:** Tiny unaligned models often **fail** red-team until SFT/RLHF + refusal data.

---

## Suggested order

```text
pretrain → fine-tune (SFT) → distill (optional compress) → red-team eval → RLHF if needed
```

Related: `REINFORCEMENT_LEARNING.md`, `TRAINING_PROCESS_COMPANY_STYLE.md`, `E2E_TASKS.md`.
