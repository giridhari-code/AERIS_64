# SENTINEL training pipeline — pretrain se end tak

Package: `sentinel` · Script: `scripts/train/train_company.py` · CLI: `sentinel-train` (optional)

---

## Big picture

```text
0  Raw text
1  Clean / dataset
2  Pretraining          ← next-token language
3  Post-training (SFT)  ← User / Assistant chat + reasoning
4  (Optional) align     ← DPO / RLHF scripts if you use them
5  Eval                 ← scores, suite
6  Serve / apps/web     ← use the checkpoint
```

---

## 0) Environment

```bash
cd ~/AERIS_64
git pull origin main
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,serve,data]"   # or: pip install torch safetensors fastapi uvicorn pyyaml tqdm
export PYTHONPATH=src
```

GPU:
```bash
# --device cuda when available
```

---

## 1) Data — clean / layout

| Stage | Format | Example path |
|-------|--------|----------------|
| Pretrain | plain `.txt` (or multi-format loader) | `data/samples/experiment_pretrain.txt` |
| SFT chat | `User:` / `Assistant:` lines | `data/samples/sft_experiment.txt` |
| Reasoning SFT | steps + `Answer:` | `data/samples/sft_reasoning.txt` |

**Pretrain text:** many sentences/paragraphs, no chat tags required.

**SFT text:**
```text
User: hello
Assistant: Hello! How can I help?

User: What is 3+4?
Assistant: 3+4=7. Answer: 7
```

Clean tips: UTF-8, drop empty lines spam, dedupe if possible, no huge binary junk.

Grow data: MB+ pretrain, hundreds–thousands SFT pairs for usable chat.

---

## 2) Pretraining (language)

Goal: model learns “next token” on your corpus.

```bash
PYTHONPATH=src python scripts/train/train_company.py   --data data/samples/experiment_pretrain.txt   --preset tiny   --steps 500   --batch-size 4   --seq-len 64   --device cpu   --out docs/runs/pretrain_v1
```

Useful flags:

| Flag | Meaning |
|------|---------|
| `--preset tiny\|small\|`… | model size |
| `--steps` | optimizer steps |
| `--batch-size` | sequences per step |
| `--seq-len` | tokens per sequence |
| `--chunk-size N` | optional chunked forward (state carry) |
| `--lr` / `--warmup` / `--cosine` | schedule |
| `--device cpu\|cuda` | device |

**Out folder:** `docs/runs/pretrain_v1/` → weights + config + tokenizer meta.

Expect: train loss drops; val may stay high if data is tiny (overfit).

---

## 3) Post-training — SFT (assistant behaviour)

Same trainer, **chat-format data**:

```bash
PYTHONPATH=src python scripts/train/train_company.py   --data data/samples/sft_experiment.txt   --preset tiny   --steps 800   --batch-size 4   --seq-len 64   --device cpu   --out docs/runs/sft_v1
```

Reasoning-style SFT:

```bash
PYTHONPATH=src python scripts/train/train_company.py   --data data/samples/sft_reasoning.txt   --preset tiny   --steps 500   --out docs/runs/reason_sft_v1   --device cpu
```

Optional continue-from: if your script supports `--checkpoint` / resume, point at `pretrain_v1` (check `--help`).

Other scripts (if you use them later):

- `train_finetune.py`, `train_align.py`, `train_rlhf.py`, `train_distill.py`

---

## 4) Optional alignment

Only after solid SFT:

- Preference pairs → `train_rlhf.py` / align scripts  
- Distill → `train_distill.py`

Skip until base SFT answers are not pure noise.

---

## 5) Eval

**Tool / SI agent (not pure LM):**
```bash
PYTHONPATH=src python scripts/eval/reasoning_score.py
PYTHONPATH=src python scripts/eval/si_v0_suite.py
PYTHONPATH=src python scripts/agent/run_si_v1.py "calculate 12*12" --json
```

**Smoke train metrics:** look at final_loss / val_nll in train log.

**Chat quality:** serve + fixed 10 questions by hand.

---

## 6) Serve (end of pipeline)

Python API:
```bash
export PYTHONPATH=src
export NEUROFIELD_CHECKPOINT=docs/runs/sft_v1   # or pretrain_v1 / reason_sft_v1
uvicorn "sentinel.serving.server:create_app" --factory --host 0.0.0.0 --port 8000
```

Frontend (`apps/web`):
```bash
cd apps/web
bun install && bun run dev
# http://localhost:3000
```

Agent without UI:
```bash
PYTHONPATH=src python scripts/agent/run_agent.py "calculate 9*9" --json
PYTHONPATH=src python scripts/agent/run_si_v1.py "what time is it"
```

---

## Minimal full path (one day, small machine)

```bash
# A. Pretrain
PYTHONPATH=src python scripts/train/train_company.py   --data data/samples/experiment_pretrain.txt   --preset tiny --steps 300 --out docs/runs/pretrain_v1 --device cpu

# B. SFT
PYTHONPATH=src python scripts/train/train_company.py   --data data/samples/sft_reasoning.txt   --preset tiny --steps 400 --out docs/runs/sft_v1 --device cpu

# C. Eval tools
PYTHONPATH=src python scripts/eval/reasoning_score.py

# D. Serve
NEUROFIELD_CHECKPOINT=docs/runs/sft_v1 PYTHONPATH=src   uvicorn "sentinel.serving.server:create_app" --factory --port 8000
```

---

## Honest limits

| Stage | Without enough data |
|-------|---------------------|
| Pretrain | memorizes tiny file; chat = gibberish |
| SFT | still weak if < hundreds of pairs |
| Reasoning claim | need model-only eval scores, not only tool 100% |

Architecture (SENTINEL) stays the same across all stages — **data and steps** change behaviour.

---

## Flag cheat sheet

```bash
PYTHONPATH=src python scripts/train/train_company.py --help
```

| You want | Set |
|----------|-----|
| Faster iterate | `--preset tiny --steps 100` |
| Better (if RAM/GPU) | `--preset small --steps 2000 --device cuda` |
| Long context chunks | `--chunk-size 16` |
| Output dir | `--out docs/runs/NAME` |

---

## Order reminder

```text
raw → clean → PRETRAIN → SFT → (align) → EVAL → SERVE
```

Do not skip straight to “reasoning claims” before SFT + scores.
