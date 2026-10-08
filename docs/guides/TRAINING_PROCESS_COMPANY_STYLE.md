# Training process — company-style (xAI / OpenAI / Google class) mapped to AERIS

**Contact:** giriisdev@gmail.com

This document explains **how frontier labs structure training** and how to run the **same stages** on AERIS at small scale.
It does **not** claim AERIS equals Grok/GPT/Claude capability.

---

## 1. Big picture (every top lab)

```text
DATA
  → PRETRAIN (next-token)
  → SFT (assistant demos)
  → PREFERENCE data (chosen vs rejected)
  → RL / DPO / PPO (align)
  → EVAL
  → ONE official checkpoint to serve
```

xAI, OpenAI, Google, Anthropic all follow this **family** of stages.
Exact data mixes, sizes, and algorithms differ and are partly secret.
**Structure is public knowledge; scale is not free.**

---

## 2. AERIS mapping

| Company stage | AERIS |
|---------------|--------|
| Data collection / filter | `data/` — only text you have rights to use |
| Pretrain | `scripts/train_company.py` |
| SFT | More User/Assistant data + same script or `train_align.py` |
| Preferences | `data/align/preferences.jsonl` |
| RLHF / DPO / PPO | `scripts/train_rlhf.py` |
| Light REINFORCE | `scripts/train_rl.py` (optional) |
| Eval | Manual chat + `scripts/daily_eval.py` |
| Official weights | `docs/AERIS_main/` (see MULTI_MACHINE_TRAINING.md) |
| Serve | `uvicorn` + `NEUROFIELD_CHECKPOINT` |

---

## 3. Rules that match “top company” discipline

1. **Data before width** — more clean text beats random larger preset on empty data.
2. **One main checkpoint** — experiments live in `docs/runs/`; promote only winners.
3. **Resume chain** — Colab/PC continue from main; do not fork three mains.
4. **Eval before promote** — if chat is garbage, do not replace main.
5. **Align after a stable base** — RLHF on a collapsed model makes it worse.
6. **Log every run** — steps, data list, preset, machine (`meta.json` + `tee` log).
7. **Legal data only** — see `DATA_POLICY.md`; contact giriisdev@gmail.com for removal.

---

## 4. Seven-day calendar (copy-paste)

Adjust steps/preset to your GPU/CPU. Names are examples.

### Day 1 — Data only

- Write or collect legal text into `data/day1.txt` (English and/or Hinglish Q&A).
- No need to train yet if data is still empty.
- Target: 50–200 clear Q&A pairs minimum for a tiny specialist.

### Day 2 — Pretrain base

```bash
git pull origin main

PYTHONPATH=src python scripts/train_company.py \
  --data data/day1.txt \
  --preset small \
  --tokenizer char \
  --steps 2000 \
  --batch-size 8 \
  --grad-accum 4 \
  --warmup 100 \
  --cosine \
  --lr 1e-3 \
  --out docs/runs/day2_pretrain \
  2>&1 | tee docs/logs/day2_pretrain.txt
```

Chat test: `hello`, `who are you`.  
If OK:

```bash
rm -rf docs/AERIS_main
cp -a docs/runs/day2_pretrain docs/AERIS_main
```

### Day 3 — More data + continue

```bash
PYTHONPATH=src python scripts/train_company.py \
  --resume docs/AERIS_main \
  --data data/day1.txt data/day3_extra.txt \
  --preset small \
  --steps 1500 \
  --grad-accum 4 --warmup 50 --cosine \
  --out docs/runs/day3_continue \
  2>&1 | tee docs/logs/day3_continue.txt
```

Promote only if chat is better or equal.

### Day 4 — SFT-style (assistant format)

Data example:

```text
User: hello
Assistant: Hello. I am AERIS. How can I help you?
```

```bash
PYTHONPATH=src python scripts/train_company.py \
  --resume docs/AERIS_main \
  --data data/sft_assistant.txt \
  --steps 800 \
  --out docs/runs/day4_sft
```

Optional:

```bash
PYTHONPATH=src python scripts/train_align.py \
  --resume docs/AERIS_main \
  --steps-sft 200 --steps-pref 100 \
  --out docs/runs/day4_align
```

### Day 5 — Preferences + RLHF

Edit `data/align/preferences.jsonl` (chosen = good, rejected = garbage).

```bash
PYTHONPATH=src python scripts/train_rlhf.py \
  --resume docs/AERIS_main \
  --prefs data/align/preferences.jsonl \
  --prompts data/rl_prompts.txt \
  --stages reward,dpo,ppo \
  --reward-steps 150 --dpo-steps 150 --ppo-steps 100 \
  --out docs/runs/day5_rlhf
```

If chat **worse**, keep old main (do not promote).

### Day 6 — Eval + scale decision

```bash
PYTHONPATH=src python scripts/daily_eval.py --checkpoint docs/AERIS_main

# If stable and you have GPU headroom:
PYTHONPATH=src python scripts/train_company.py \
  --resume docs/AERIS_main \
  --preset medium \
  --data data/day1.txt data/day3_extra.txt data/sft_assistant.txt \
  --steps 3000 --grad-accum 8 --warmup 200 --cosine --bf16 \
  --out docs/runs/day6_medium
```

Note: changing preset/size may not load old weights cleanly — treat as new family if shapes differ.

### Day 7 — Serve + document

```bash
NEUROFIELD_CHECKPOINT=docs/AERIS_main PYTHONPATH=src \
  uvicorn "neurofield.serving.server:create_app" --factory --host 0.0.0.0 --port 8000
```

Write in `docs/logs/week1_notes.txt`: what data, what worked, what failed.

---

## 5. Serve command (always)

```bash
NEUROFIELD_CHECKPOINT=docs/AERIS_main PYTHONPATH=src \
  uvicorn "neurofield.serving.server:create_app" --factory --host 0.0.0.0 --port 8000
```

UI: temp `0.2`, max tokens cap `64` for tiny/small models (not 2048).

---

## 6. Multi-person / Colab

See **[MULTI_MACHINE_TRAINING.md](MULTI_MACHINE_TRAINING.md)**.

- Train → `docs/runs/...`
- Promote → `docs/AERIS_main`
- Others `--resume docs/AERIS_main`
- Never delete all `runs/` history if you need rollback

---

## 7. Scale ladder

| Preset | When |
|--------|------|
| `tiny` | Pipeline test |
| `small` | First real specialist |
| `medium` / `large` | GPU + more data |
| `xl` / `aeris_1b.yaml` | Serious GPU + real corpus |

Details: **[SCALE_AND_OPTIMIZE.md](SCALE_AND_OPTIMIZE.md)**, **[TRAIN_1B.md](TRAIN_1B.md)**.

---

## 8. Honest limits

| Company | AERIS on a laptop |
|---------|-------------------|
| Billions of parameters | Thousands–millions |
| Trillions of tokens | Thousands of characters unless you add data |
| Cluster training | One PC / one Colab GPU |
| Same **stage names** | Yes |
| Same **intelligence** | No |

Garbage chat almost always means: weak data, wrong checkpoint, or train too short — not “missing one magic flag”.

---

## 9. Related docs

| Doc | Topic |
|-----|--------|
| [RUN_AND_TRAIN.md](RUN_AND_TRAIN.md) | Basic commands |
| [MULTI_MACHINE_TRAINING.md](MULTI_MACHINE_TRAINING.md) | PC + Colab weights |
| [REINFORCEMENT_LEARNING.md](REINFORCEMENT_LEARNING.md) | RM / DPO / PPO |
| [SCALE_AND_OPTIMIZE.md](SCALE_AND_OPTIMIZE.md) | Cosine, accum, presets |
| [DATA_POLICY.md](../DATA_POLICY.md) | Legal / removal |
| [MEMORY_FIXES.md](MEMORY_FIXES.md) | Architecture fixes |

---

## 10. Checklist before you say “trained like a top company”

- [ ] Clean legal data in `data/`
- [ ] Pretrain run logged under `docs/runs/`
- [ ] Chat smoke test passed
- [ ] `docs/AERIS_main` updated only after test
- [ ] Optional SFT + preference + RLHF
- [ ] Eval notes saved
- [ ] Serve uses `AERIS_main`, not a random old folder

That **is** the process. Scale is a separate budget problem.
