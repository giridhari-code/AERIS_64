# AERIS / NeuroField — Training 0 se 100

Ek hi guide: bilkul shuru se trained model + chat tak.

Contact: giriisdev@gmail.com  
Repo: https://github.com/giridhari-code/AERIS_64

---

## Badi picture (yaad rakhne layak)

```text
0  Setup
1  Raw text
2  Clean → data/ready/
3  Pretrain (raw text)
4  Test
5  SFT (User / Assistant)     ← optional but recommended
6  Eval
7  Promote → AERIS_main
8  Serve (chat)
9  Tools / red-team / more     ← later
100 Done — improve data & repeat
```

---

## Level 0 — Setup (ek baar)

### Local

```bash
git clone https://github.com/giridhari-code/AERIS_64.git
cd AERIS_64
git pull origin main

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -U pip
pip install torch safetensors fastapi uvicorn pydantic numpy
```

### Colab

1. Runtime → GPU (optional but faster)
2. Cells:

```python
!git clone https://github.com/giridhari-code/AERIS_64.git
%cd AERIS_64
!pip install -q torch safetensors fastapi uvicorn pydantic numpy
```

---

## Level 1 — Raw text

Apna topic likho ya paste karo (paper, notes, FAQ).

```bash
mkdir -p data/raw data/ready docs/runs
nano data/raw/my_raw.txt
```

Abhi **User / Assistant mat likho**. Sirf normal paragraphs.

---

## Level 2 — Clean

```bash
python3 << 'PY'
from pathlib import Path
import re
raw = Path("data/raw/my_raw.txt").read_text(encoding="utf-8", errors="ignore")
text = re.sub(r"<[^>]+>", " ", raw)
text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
text = re.sub(r"[ \t]+", " ", text)
lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
Path("data/ready/pretrain_raw.txt").write_text("\n\n".join(lines), encoding="utf-8")
print("OK data/ready/pretrain_raw.txt", len(lines), "lines")
PY
```

Check:

```bash
head -20 data/ready/pretrain_raw.txt
```

---

## Level 3 — Pretrain (raw text only)

**Matlab:** model seekhe “agla word kya hai”.

### Simple command (bas itna)

```bash
PYTHONPATH=src python scripts/train/train_company.py \
  --data data/ready/pretrain_raw.txt \
  --preset small \
  --steps 1500 \
  --out docs/runs/pretrain_v1 \
  --device cpu
```

GPU / Colab:

```bash
  --device cuda
```

### Optional stronger flags (baad mein)

```text
--batch-size 4
--grad-accum 4
--warmup 100
--cosine
--lr 1e-3
```

### Output

```text
docs/runs/pretrain_v1/
  model.safetensors
  config / tokenizer files
```

Loss logs mein dheere kam hona chahiye.

---

## Level 4 — Pehla test

```bash
NEUROFIELD_CHECKPOINT=docs/runs/pretrain_v1 PYTHONPATH=src \
  uvicorn "neurofield.serving.server:create_app" --factory --host 0.0.0.0 --port 8000
```

Browser: `http://127.0.0.1:8000`  
Ya API se short prompt bhejo.

**Expect:** abhi perfect assistant nahi — sirf thoda language/topic feel.

---

## Level 5 — SFT (chat style) — recommended

### 5a Q&A file

```bash
nano data/ready/sft_chat.txt
```

Format:

```text
User: What is this about?
Assistant: Short clear answer.

User: Sawal Hindi mein?
Assistant: Seedha jawab.
```

### 5b Fine-tune from pretrain

```bash
PYTHONPATH=src python scripts/train/train_finetune.py \
  --resume docs/runs/pretrain_v1 \
  --data data/ready/sft_chat.txt \
  --steps 800 \
  --preset small \
  --out docs/runs/sft_v1 \
  --device cpu
```

Colab: `--device cuda`.

---

## Level 6 — Eval

```bash
# full path tasks
PYTHONPATH=src python scripts/eval/run_e2e_tasks.py \
  --checkpoint docs/runs/sft_v1 \
  --tasks data/e2e_tasks/full_tasks.jsonl \
  --device cpu

# safety probes (optional)
PYTHONPATH=src python scripts/eval/red_team.py \
  --checkpoint docs/runs/sft_v1 \
  --device cpu
```

Manual: 5 questions khud pucho. Sense bane tab aage.

---

## Level 7 — Promote official main

```bash
# backup purana
mv docs/AERIS_main docs/AERIS_main_backup 2>/dev/null || true
cp -a docs/runs/sft_v1 docs/AERIS_main
```

Weights **git mein mat push karo** (policy). Apne disk / Drive pe rakho.

---

## Level 8 — Serve daily use

```bash
export NEUROFIELD_CHECKPOINT=docs/AERIS_main
export NEUROFIELD_WEB_TOOLS=1
export PYTHONPATH=src

uvicorn "neurofield.serving.server:create_app" --factory --host 0.0.0.0 --port 8000
```

Tools (weather, search, time, calc, wiki, FX) prompt se auto chal sakte hain agar internet ho.

---

## Level 9 — Baad ke tools (jab comfortable ho)

| Cheez | Command / doc |
|--------|----------------|
| Continue train | `--resume docs/runs/...` more `--steps` |
| Distill | `scripts/train/train_distill.py` |
| RLHF | `docs/guides/REINFORCEMENT_LEARNING.md` |
| Multi machine | `docs/guides/MULTI_MACHINE_TRAINING.md` |
| Scale | `docs/guides/SCALE_AND_OPTIMIZE.md` |
| Full E2E | `docs/guides/E2E_TASKS.md` |

---

## Level 100 — Loop (asli progress)

```text
better data → train/finetune → eval → promote → serve → user feedback → better data
```

Intelligence **data × time × eval** se badhti hai, ek magic flag se nahi.

---

## Colab short path

```python
!git clone https://github.com/giridhari-code/AERIS_64.git
%cd AERIS_64
!pip install -q torch safetensors fastapi uvicorn pydantic numpy
# upload or write data/ready/pretrain_raw.txt
!PYTHONPATH=src python scripts/train/train_company.py \
  --data data/ready/pretrain_raw.txt --preset small --steps 1500 \
  --out docs/runs/pretrain_v1 --device cuda
!cd docs/runs && zip -r /content/pretrain_v1.zip pretrain_v1
```

Download zip → local `docs/runs/` → finetune / serve.

---

## Format cheatsheet

| Stage | File content |
|--------|----------------|
| Pretrain | Plain paragraphs only |
| SFT | `User:` / `Assistant:` pairs |
| Prefer not | Full copyrighted books |

---

## Common errors

| Problem | Fix |
|---------|-----|
| `No module neurofield` | `PYTHONPATH=src` |
| CUDA OOM | `--batch-size 1 --grad-accum 8 --device cuda` or cpu |
| Empty loss / crash on data | file path + non-empty UTF-8 text |
| Garbage chat | more clean SFT data, not only 10 lines |
| Colab disconnect | zip often; `--resume` next session |

---

## Minimum path (agar sirf 1 din)

```text
0 setup → 2 small ready txt → 3 pretrain 500–1500 steps → 4 chat test
```

Phir jab time ho: 5 SFT → 7 main → 8 serve.

---

*End of Training 0 to 100. Start at Level 0; do not skip to Level 9 on day one.*
