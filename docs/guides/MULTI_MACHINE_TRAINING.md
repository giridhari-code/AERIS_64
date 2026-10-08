# Multi-machine / multi-person training (full guide)

**Contact:** giriisdev@gmail.com

When several people train (home PC, laptop, Google Colab), you get **multiple weight files**.
They do **not** merge by themselves. This doc is the full policy for AERIS / NeuroField.

---

## 1. The problem (simple)

```text
You train on PC        →  model file A
Friend trains on Colab →  model file B
Someone else trains    →  model file C
```

Three files = **three different brains**.

The chat server loads **one** checkpoint at a time:

```bash
NEUROFIELD_CHECKPOINT=docs/SOME_FOLDER ...
```

There is no automatic “combine A+B+C into one smarter model”.

---

## 2. The rule (one line)

**One official checkpoint. Everyone continues from it. Only one person updates “main” after a good run.**

```text
docs/AERIS_main/     ← single source of truth (name it what you want)
```

---

## 3. Roles

| Role | What they do |
|------|----------------|
| **Owner** | Decides which run becomes `AERIS_main`; owns git / Drive folder |
| **Trainer** | Runs train on PC or Colab with `--resume` from main, then reports results |
| **User** | Only serves / tests; does not overwrite main without owner OK |

On a solo project, you are all three roles — still keep **one main folder**.

---

## 4. Folder layout (recommended)

```text
docs/
  AERIS_main/                 # official weights (serve this)
    config.json
    model.pt  or  model.safetensors
    tokenizer.json
    vocab.json
    meta.json
  runs/
    2026-10-08_pc_small_s1000/
    2026-10-09_colab_small_s2000/
  logs/
    train_pc_2026-10-08.txt
    train_colab_2026-10-09.txt
data/
  shared/                     # data everyone may use (legal only)
  private/                    # do not commit secrets / user PII
```

Copy a finished experiment into `runs/…`, evaluate, **then** copy the winner into `AERIS_main/`.

---

## 5. Workflow A — sequential resume (best for small teams)

```text
Day 1  PC:     train from scratch or old main → upload as main
Day 2  Colab:  --resume main → more steps → upload as main
Day 3  PC:     --resume main → more steps → upload as main
```

### Commands

**PC or Colab (same idea):**

```bash
git pull origin main

PYTHONPATH=src python scripts/train_company.py \
  --resume docs/AERIS_main \
  --data data/shared/batch_day2.txt \
  --preset small \
  --steps 1000 \
  --grad-accum 4 \
  --warmup 50 \
  --cosine \
  --out docs/runs/2026-10-09_colab_small_s1000

# After chat test looks OK, promote to main:
rm -rf docs/AERIS_main
cp -a docs/runs/2026-10-09_colab_small_s1000 docs/AERIS_main
```

**Serve only main:**

```bash
NEUROFIELD_CHECKPOINT=docs/AERIS_main PYTHONPATH=src \
  uvicorn "neurofield.serving.server:create_app" --factory --host 0.0.0.0 --port 8000
```

---

## 6. Workflow B — parallel experiments (OK, but no auto-merge)

Several people may train **at the same time** for experiments:

```text
runs/exp_hindi_qa/
runs/exp_code_only/
runs/exp_rlhf_try/
```

Rules:

1. Each run writes to its **own** `docs/runs/...` folder — never overwrite `AERIS_main` mid-run.
2. Compare with the same eval prompts (hello, namaste, who are you).
3. Owner picks **one** winner → copy to `AERIS_main`.
4. Do **not** average random checkpoints unless you know what model soup is.

---

## 7. Google Colab specifics

1. Clone the same git commit as everyone else.
2. Upload or mount the `docs/AERIS_main` folder (Drive is fine).
3. Train with `--resume` pointing at that folder.
4. Download the **whole** output folder (config + weights + tokenizer + vocab + meta).
5. Owner replaces `AERIS_main` only after a quick chat test.

**Must match across machines:**

| Must be same | Why |
|--------------|-----|
| Code version (`git` commit) | Avoid silent bugs |
| `preset` / `d_model` / vocab | Load fails or garbage output |
| Tokenizer files | Token IDs must match weights |

If vocab size or architecture differs, `--resume` cannot meaningfully continue — you start a new family of weights.

---

## 8. What to put in `meta.json` (every run)

Training scripts already write meta; if you edit by hand, include:

```json
{
  "product": "AERIS_64",
  "machine": "colab-T4",
  "who": "alice",
  "date": "2026-10-09",
  "steps": 1000,
  "preset": "small",
  "data": ["data/shared/batch_day2.txt"],
  "parent_checkpoint": "docs/AERIS_main",
  "notes": "Hinglish FAQ only"
}
```

---

## 9. Sharing files

| Method | Use for |
|--------|---------|
| **Git LFS / release** | Small demos only (large weights may be too big) |
| **Google Drive / HF Hub** | Real checkpoints between PC and Colab |
| **Git (code only)** | Scripts, configs, small `data/` samples |

Do not commit huge weight files if the host rejects them; share weights via Drive/HF and keep **code + policy** in git.

---

## 10. Eval before promoting to main

Minimum chat checks:

```text
hello / hi
namaste
who are you
thank you
```

If output is garbage (`couccc`, random syllables), **do not** promote that run to `AERIS_main`.

Optional:

```bash
PYTHONPATH=src python scripts/daily_eval.py --checkpoint docs/runs/YOUR_RUN
```

---

## 11. Align / RLHF on multi-machine

Same rule:

```text
resume from AERIS_main → train_rlhf.py → new runs/... → eval → promote
```

```bash
PYTHONPATH=src python scripts/train_rlhf.py \
  --resume docs/AERIS_main \
  --prefs data/align/preferences.jsonl \
  --stages reward,dpo,ppo \
  --out docs/runs/2026-10-09_rlhf
```

If RLHF makes chat worse, keep previous main; discard the run.

---

## 12. What does *not* solve multi-weight chaos

| Idea | Reality |
|------|---------|
| Average all models always | Often worse; only for careful soup experiments |
| Everyone force-pushes main | Overwrites and confusion |
| Different presets, one resume | Architecture mismatch |
| Train without sharing tokenizer | Garbage text at serve time |

---

## 13. Minimal policy poster

```text
1. Official folder: docs/AERIS_main
2. Train → docs/runs/DATE_machine_preset_steps
3. Test chat
4. Only then copy run → AERIS_main
5. Others always --resume docs/AERIS_main
6. Same git commit + same preset + full tokenizer folder
```

---

## 14. Related docs

| Doc | Topic |
|-----|--------|
| [RUN_AND_TRAIN.md](RUN_AND_TRAIN.md) | Basic train / serve |
| [SCALE_AND_OPTIMIZE.md](SCALE_AND_OPTIMIZE.md) | Presets, cosine, accum |
| [REINFORCEMENT_LEARNING.md](REINFORCEMENT_LEARNING.md) | RLHF stages |
| [DATA_POLICY.md](../DATA_POLICY.md) | Legal data / removal |
| [TRAIN_1B.md](TRAIN_1B.md) | Large GPU recipe |

---

## 15. Disclaimer

This is an engineering coordination guide for a personal research repo.
It is not a substitute for formal MLOps or legal advice for a commercial product.
