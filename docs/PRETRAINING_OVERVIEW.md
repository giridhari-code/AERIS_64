# Pretraining Overview

> Based on standard LLM pretraining curriculum (e.g. Stanford-style overview):  
> **Goal → objective → data mixtures.**

---

## 1. Goal

**Learn patterns of language and code.**

Model ko itna expose karo ki woh:

- natural language ke structures (grammar, facts, style)
- code ke structures (syntax, APIs, common patterns)

dono seekh le — bina har task ke liye alag label ke.

NeuroField Company mein yahi goal chhote scale pe: identity, Hindlish, skills, API help, thoda code.

---

## 2. Objective function

**Predict the next token.**

Training step:

```text
input:  tokens[0 .. t]
target: tokens[t+1]
loss:   cross-entropy( model_logits, target )
```

Ye **self-supervised** hai:

- alag human labels ki zaroorat nahi
- bas raw text / code stream

NeuroField train scripts (`train_company.py`, `train_multilang.py`) yahi karte hain:

- character-level **ya** BPE tokens
- next-token CE (+ optional error-energy / balance terms from architecture)

---

## 3. Data mixtures

Frontier / Stanford-style mix roughly:

| Source type | Examples | Role |
|-------------|----------|------|
| **Web-scraped** | Common Crawl, Wikipedia | Broad language, world knowledge |
| **Code** | GitHub, Stack Overflow | Syntax, algorithms, APIs |

Practical mixing ideas:

```text
mixture ≈  web text  +  encyclopedia  +  code  +  (optional) books/papers
```

Ratios labs change karti hain (e.g. more code → better coding; more curated wiki → cleaner facts).

### Is repo mein corresponding files

| Mixture idea | Local stand-in |
|--------------|----------------|
| Web / general language | `data/company_corpus.txt`, `data/india_multilang.txt` |
| “Wikipedia-like” curated | skills + short factual lines you add |
| Code (GitHub / SO style) | Python snippets in corpus + apna `data/code_corpus.txt` |

**Note:** Common Crawl / full GitHub dump yahan ship nahi hote (size + license).  
Track B (frontier) pe aap khud legal dumps + filters use karoge.

---

## 4. How this maps to NeuroField daily training

```text
Stanford concept          →  Aapka action
─────────────────────────    ──────────────────────────
Goal: language + code     →  text + code files in data/
Next-token objective      →  train_company.py CE loss
Web-scraped mixture       →  expand company / multilang text
Code mixture              →  add GitHub-style functions, SO-like Q&A
```

Example Day plan:

```bash
# 1) grow mixture
echo "def fib(n): ..." >> data/code_corpus.txt
echo "Wikipedia-style paragraph..." >> data/company_corpus.txt

# 2) next-token pretrain step
PYTHONPATH=src python scripts/train_company.py \
  --data data/company_corpus.txt data/code_corpus.txt data/india_multilang.txt \
  --steps 800 \
  --out docs/neurofield_company
```

---

## 5. Minimal “pretrain mixture” checklist

- [ ] General language text (chat, FAQ, articles)
- [ ] Some encyclopedic / factual lines
- [ ] Code files or code-heavy lines
- [ ] Optional: Hindlish / Hindi for India
- [ ] Dedup + remove pure garbage lines
- [ ] Eval after each mixture change

---

## 6. Scale reminder

| Scale | Data mixture size (order) |
|-------|---------------------------|
| This zip (toy) | thousands of characters / tokens |
| Small product | millions–billions tokens |
| Frontier | often **trillions** of tokens from web + code (+ more) |

Same **goal + next-token + mixtures** — difference mostly **volume, quality filters, and compute**.

---

## Related

- `docs/FRONTIER_TRAINING_ROADMAP.md` — Day 1…N plan  
- `docs/RUN_AND_TRAIN.md` — commands  
- `docs/COMPANY_STATUS.md` — what this repo is / is not  


---

## Full Common Crawl

**Not included in this repository** (multi-tiabyte / petabyte archive).

See **[docs/COMMON_CRAWL.md](COMMON_CRAWL.md)** for:

- official access (`data.commoncrawl.org`)
- why labs use **filtered** derivatives (FineWeb, FineWeb-Edu)
- safe sample workflow (`scripts/fetch_cc_sample.py`)
