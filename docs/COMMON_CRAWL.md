# Full Common Crawl — Reality, Access, and What to Use Instead

## Short answer

**Full Common Crawl is not a single download you put in this zip.**

| Fact | Detail |
|------|--------|
| What it is | Non-profit **monthly** snapshots of the public web |
| One recent crawl (Sep 2026, CC-MAIN-2026-39) | ~**2.17 billion pages**, ~**100+ TiB compressed** (WARC much larger) |
| Many years of crawls | Cumulative **petabyte-class** archive |
| Cost to store “everything” | Enterprise disk + network; not a laptop |
| License / access | Free to access via AWS Open Data / HTTP |

So: Stanford slide says “web-scraped (e.g. Common Crawl)” as a **mixture source** — labs **sample + filter** it; they do not “copy full CC into `data/`”.

---

## Official access

- Get started: https://commoncrawl.org/get-started  
- HTTP base: `https://data.commoncrawl.org/`  
- S3: `s3://commoncrawl/` (best from **AWS us-east-1**)

### File types

| Format | Contents |
|--------|----------|
| **WARC** | Raw crawl (HTML, headers) — largest |
| **WAT** | Metadata |
| **WET** | Extracted plain text (easier for LM experiments) |

Example paths file for one crawl:

```text
https://data.commoncrawl.org/crawl-data/CC-MAIN-2026-39/wet.paths.gz
```

Each line in `wet.paths.gz` is a relative path to one WET gzip part.  
A single crawl can have on the order of **~100,000** WET files.

---

## What frontier / open LLM teams actually use

Almost nobody trains on **raw unfiltered full CC** as-is.

Typical pipeline:

```text
Common Crawl (WARC/WET)
    → text extraction
    → language ID (keep en / hi / …)
    → spam / adult / boilerplate filters
    → deduplication (MinHash etc.)
    → optional quality / educational classifier
    → tokenized shards for training
```

### Ready-made CC derivatives (recommended)

| Dataset | What | Scale (order) | Notes |
|---------|------|----------------|-------|
| **FineWeb** | Filtered CC → English web | ~15T tokens | Hugging Face; strong open baseline |
| **FineWeb-Edu** | Quality-filtered subset | ~1.3T+ | Often better than raw FineWeb at fixed compute |
| **FineWeb-2** | Multilingual filtered web | large | 1000+ languages |
| C4 / RefinedWeb / Dolma | Older / other recipes | various | Still used in papers |

FineWeb samples (practical download sizes):

| Config | Approx size |
|--------|-------------|
| `sample-10BT` | ~tens of GB |
| `sample-100BT` | ~hundreds of GB |
| `sample-350BT` | larger |
| full FineWeb | tens of TB |

Example (on **your** machine with disk + bandwidth):

```python
# pip install datasets
from datasets import load_dataset

# stream — do not download all at once
ds = load_dataset(
    "HuggingFaceFW/fineweb",
    name="sample-10BT",   # start small
    split="train",
    streaming=True,
)

for i, row in enumerate(ds):
    text = row.get("text") or ""
    # write to your shards / train pipeline
    if i >= 1000:
        break
```

Or snapshot download with patterns — see FineWeb dataset card on Hugging Face.

---

## NeuroField Company (this repo)

| Can we ship full CC in the zip? | **No** |
|--------------------------------|--------|
| Local stand-in | `data/company_corpus.txt`, `india_multilang.txt`, `code_corpus.txt` |
| Next step on a real server | Stream FineWeb `sample-10BT` → clean → `train_company.py` / larger trainer |
| Script stub | `scripts/fetch_cc_sample.py` (documents workflow; does not pull petabytes) |

**Do not** try:

```bash
wget -r https://data.commoncrawl.org/   # will not finish sensibly on a laptop
```

---

## Realistic ladder

```text
Day 1–30     Local text mixtures (this repo)
Month 2+     FineWeb sample-10BT (stream or partial download)
Later        sample-100BT / FineWeb-Edu on multi-GPU
Lab scale    Many CC snapshots + own filters + code mix (The Stack, etc.)
```

Same Stanford idea:

- **Goal:** language (+ code) patterns  
- **Objective:** next-token prediction  
- **Mixture:** web (CC-derived) + code  

Difference = **how much filtered web you can store and train on**.

---

## Related docs

- `docs/PRETRAINING_OVERVIEW.md` — goal / next-token / mixtures  
- `docs/FRONTIER_TRAINING_ROADMAP.md` — Day N plan  
- `docs/COMPANY_STATUS.md` — scope limits  

### References (public)

- Common Crawl get started  
- FineWeb / FineWeb-Edu on Hugging Face (`HuggingFaceFW/fineweb`)  
