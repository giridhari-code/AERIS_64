# AERIS_64 / SENTINEL — Full Training Pipeline

Yeh document **poori training pipeline** explain karta hai: pre-train → mid → long context → SFT → reasoning → preference / DPO / RLHF, data examples ke saath.

Official entrypoints:

```bash
# Single-stage LM (pretrain-style bins)
python -m sentinel.cli train --config configs/colab.yaml --data-dir data/tokens

# Multi-stage pipeline
python -m sentinel.cli pipeline --stages pretrain,midtrain,sft --config configs/colab.yaml ...
python -m sentinel.cli pipeline --list-stages
```

Default stage order (`sentinel.pipeline.runner.DEFAULT_ORDER`):

```text
pretrain → midtrain → long_context → sft → reasoning
  → preference_bt → dpo → orpo → lora_pref → grpo → rlhf_ppo → opd
```

Chhote Colab runs ke liye aksar sirf: **pretrain → midtrain → sft** kaafi hai.

---

## Big picture

| Stage | Data style | Model kya seekhta hai |
|--------|------------|------------------------|
| **Pre-train** | Raw text (books, web, notes) | Language: “agli token kya?” |
| **Mid-train** | Better / domain raw text | Same, higher quality |
| **Long context** | Lambi sequences | Lamba context handle |
| **SFT** | `User:` / `Assistant:` | Instructions follow, chat style |
| **Reasoning** | CoT / step-by-step demos | Soch ke jawab |
| **Preference / DPO / RLHF** | chosen vs rejected | Insaan-pasand replies |

Architecture (SENTINEL) har stage pe wahi rehti hai: two-speed memory, field, skills, hierarchical predict. Farq **data + LR + steps** ka hai.

---

## 0) Shared setup

```bash
git clone https://github.com/giridhari-code/AERIS_64.git
cd AERIS_64
pip install -e ".[dev]"
export PYTHONPATH=packages
```

Tokenizer ek baar banao; **baad ke stages mein same `tokenizer.json` reuse karo**.

```bash
python scripts/data/prepare_data.py \
  --input "data/raw/*.txt" \
  --out-dir data/tokens \
  --vocab-size 8000 \
  --val-frac 0.02 \
  --chunk-chars 2000
```

Outputs: `data/tokens/train.bin`, `val.bin`, `tokenizer.json`.

---

## 1) Pre-training (raw text)

### Kya hota hai

Model ko **bahut saara normal text** dikhaya jata hai. Har step pe:

```text
context tokens  →  model  →  next-token prediction (CE loss)
```

Sath mein AERIS extras: hierarchical prediction errors, memory writes, skill routing (training config ke hisaab se).

### Data example — **raw text** (`.txt`)

```text
The monsoon arrived early that year. Farmers watched the sky and waited.
In the evening the streets filled with the smell of wet earth.

भारत एक विविधतापूर्ण देश है। यहाँ कई भाषाएँ और संस्कृतियाँ हैं।
Science is the careful study of the natural world through observation and experiment.
```

- Paragraphs / lines — koi `User:` zaroori nahi  
- Hinglish / Hindi / English mix OK  
- Zyada clean, diverse text = behtar pre-train  

File rakhne ki jagah: `data/raw/*.txt`

### Commands

```bash
# tokenize
python scripts/data/prepare_data.py \
  --input "data/raw/*.txt" \
  --out-dir data/tokens \
  --vocab-size 8000 \
  --val-frac 0.02 \
  --chunk-chars 2000

# train (Colab-size)
python -m sentinel.cli train \
  --config configs/colab.yaml \
  --data-dir data/tokens \
  --tokenizer-json data/tokens/tokenizer.json \
  --max-steps 5000
```

Ya pipeline:

```bash
python -m sentinel.cli pipeline \
  --stages pretrain \
  --config configs/colab.yaml \
  --data-dir data/tokens \
  --tokenizer-json data/tokens/tokenizer.json \
  --output-dir runs/pipeline
```

### Checkpoints

- `runs/colab/checkpoint_latest.pt` / `checkpoint_best.pt`  
- Export folder: `runs/colab/final/` (`model.safetensors` + config)  

Serve **sirf safetensors folder** se (`.pt` seedha load nahi).

---

## 2) Mid-training

### Kya hota hai

Pre-train ke baad **behtar / domain text** pe continue (lower LR). Architecture same; objective mostly next-token.

### Data

Phir se **raw text**, lekin:

- Domain (code docs, support FAQs, product text, …)  
- Ya simply **zyada + cleaner** corpus  

### Command

```bash
python -m sentinel.cli train \
  --config configs/colab.yaml \
  --data-dir data/tokens \
  --tokenizer-json data/tokens/tokenizer.json \
  --resume \
  --lr 2e-4 \
  --max-steps 10000
```

Pipeline stage `midtrain` default: LR ≈ `0.3 ×` base, steps capped.

---

## 3) Long context (optional)

Seq length badhakar (e.g. 256–512+) train; batch chhota, LR aur kam.

```bash
python -m sentinel.cli pipeline \
  --stages long_context \
  --config configs/colab.yaml \
  --data-dir data/tokens \
  --tokenizer-json data/tokens/tokenizer.json \
  --resume runs/colab/checkpoint_latest.pt \
  --output-dir runs/pipeline
```

---

## 4) SFT — Supervised fine-tuning (post-train chat)

### Kya hota hai

Model ko **dialogue demos** dikhao taaki woh `User` ke baad `Assistant` style jawab de.

Yeh **post-training** ka main hissa hai chhote setups mein.

### Data example — **User / Assistant**

```text
User: hello
Assistant: Hello. I am AERIS_64. How can I help you?

User: who are you
Assistant: I am AERIS_64, a small AI assistant. I try to be helpful and concise.

User: what is 2+2
Assistant: 2 + 2 = 4.

User: namaste, mausam kaisa hai baat karne ke liye
Assistant: Namaste. Main mausam API use nahi karta; aap local forecast dekh sakte ho. Aur kaise madad karun?
```

Repo samples:

- `data/align/sft_demos.txt`  
- `data/align/sft_specials.txt`  
- `data/demo/sft.txt`  

Supported formats (corpus loader): plain `User:`/`Assistant:`, ChatML JSONL, Alpaca `instruction`/`output`, prompt/response, …

### Commands

```bash
# Option A — pipeline SFT (recommended)
python -m sentinel.cli pipeline \
  --stages sft \
  --config configs/colab.yaml \
  --sft-data data/align/sft_demos.txt \
  --tokenizer-json data/tokens/tokenizer.json \
  --resume runs/colab/checkpoint_latest.pt \
  --output-dir runs/sft

# Option B — pack SFT text to bins then train (lower LR)
python scripts/data/prepare_data.py \
  --input "data/align/*.txt" \
  --out-dir data/tokens_sft \
  --vocab-size 8000 \
  --val-frac 0.05 \
  --chunk-chars 1500

python -m sentinel.cli train \
  --config configs/colab.yaml \
  --data-dir data/tokens_sft \
  --tokenizer-json data/tokens/tokenizer.json \
  --resume \
  --lr 1e-4 \
  --max-steps 2000
```

**Tips**

- Same tokenizer as pre-train  
- LR chhota (`1e-4` class)  
- Steps kam (1k–3k) — warna pre-train forget  
- Success = chat style, sirf CE nahi  

---

## 5) Reasoning (optional)

SFT jaisa loop, data chain-of-thought / step-by-step:

```text
User: If a train goes 60 km/h for 2 hours, how far does it go?
Assistant: Speed is 60 km/h and time is 2 h. Distance = speed × time = 120 km.
```

```bash
python -m sentinel.cli pipeline \
  --stages reasoning \
  --config configs/colab.yaml \
  --reasoning-data data/samples/sft_reasoning.txt \
  --tokenizer-json data/tokens/tokenizer.json \
  --resume runs/sft/checkpoint_latest.pt \
  --output-dir runs/reasoning
```

---

## 6) Preference stages (advanced)

Jab do jawab mein se **better** sikhana ho.

### Preference JSONL example

```json
{"prompt": "User: say hi\nAssistant:", "chosen": "Hi! How can I help?", "rejected": "asdkjashd garbage"}
```

| Stage | Idea |
|--------|------|
| **preference_bt** | Reward model: P(chosen ≻ rejected) |
| **dpo** | Direct preference optimization |
| **orpo** | Odds-ratio preference |
| **lora_pref** | Preference with LoRA adapters |
| **grpo** | Group relative policy style |
| **rlhf_ppo** | Classic RLHF + PPO |
| **opd** | Online / further preference (repo stage) |

```bash
python -m sentinel.cli pipeline \
  --stages dpo \
  --config configs/colab.yaml \
  --prefs data/prefs/prefs.jsonl \
  --tokenizer-json data/tokens/tokenizer.json \
  --resume runs/sft/checkpoint_latest.pt \
  --output-dir runs/dpo
```

Chhote Colab models pe yeh stages **optional**; pehle solid **SFT** karo.

---

## 7) Full pipeline one-liner (ambitious)

```bash
python -m sentinel.cli pipeline \
  --stages pretrain,midtrain,sft \
  --config configs/colab.yaml \
  --data-dir data/tokens \
  --sft-data data/align/sft_demos.txt \
  --tokenizer-json data/tokens/tokenizer.json \
  --output-dir runs/full
```

Poora DEFAULT_ORDER tab jab data + GPU budget dono maujood hon.

---

## 8) After training — serve (model-only)

```bash
export NEUROFIELD_CHECKPOINT=runs/colab/final   # or runs/sft export folder
export PYTHONPATH=packages
uvicorn sentinel.api.app:create_app --factory --host 127.0.0.1 --port 8000
```

```bash
curl -s http://127.0.0.1:8000/readyz
curl -s http://127.0.0.1:8000/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Namaste","max_new_tokens":64}'
```

API model-only hai: tools / agent / RAG routes hata diye gaye hain.

---

## 9) Practical size guide (Colab)

| Item | Rough |
|------|--------|
| Pre-train tokens | 200k minimum smoke; **1M+** better |
| Model (colab.yaml) | ~1.5M params (`d_model=128`) — chhota |
| Pre-train steps | 3k–15k |
| SFT steps | 1k–3k |
| Train CE goal | neeche **~4**, phir **~3.5** |
| Val CE | train ke qareeb; **30+ = problem** (data/tokenizer/eval) |

---

## 10) Data format cheat sheet

**Pre-train / mid (raw):**

```text
Plain paragraphs. No roles required.
हिंदी और English दोनों ठीक हैं.
```

**SFT (dialogue):**

```text
User: <question>
Assistant: <answer>
```

**Preference:**

```json
{"prompt": "...", "chosen": "...", "rejected": "..."}
```

**Alpaca JSONL (also supported by corpus loader):**

```json
{"instruction": "Translate to Hindi", "input": "Hello", "output": "नमस्ते"}
```

---

## 11) Mental model (one diagram)

```text
raw .txt  ──prepare_data──► train.bin + tokenizer.json
                              │
                              ▼
                         PRETRAIN (next token)
                              │ resume
                              ▼
                         MIDTRAIN (better raw, lower LR)
                              │ optional
                              ▼
                         LONG_CONTEXT
                              │
         User/Assistant .txt ──┤
                              ▼
                            SFT (chat behaviour)
                              │ optional
                              ▼
                      reasoning / DPO / RLHF
                              │
                              ▼
                      export final/ → uvicorn /v1/chat
```

---

## 12) Common mistakes

1. SFT pe **naya tokenizer** bana dena → embeddings mismatch  
2. Sirf 200k tokens pe “full intelligence” expect karna  
3. Serve pe **`.pt` folder** dena bina `model.safetensors`  
4. SFT ke bina chat style expect karna (pre-train raw text hai)  
5. Val CE ignore karke sirf train loss dekhna  

---

*AERIS_64 / SENTINEL — training pipeline reference. Code: `packages/sentinel/pipeline/`, `scripts/data/prepare_data.py`, `configs/colab.yaml`.*
