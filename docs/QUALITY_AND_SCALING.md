# Quality = data + size + align (not max_tokens alone)

## What actually makes answers good

| Pillar | Meaning | AERIS_64 today | Top models (Claude / GPT / Grok class) |
|--------|---------|----------------|------------------------------------------|
| **Data** | How much clean text the model saw | Thousands of characters | Billions–trillions of tokens |
| **Size** | Parameters / capacity | Tiny (~10⁴–10⁵ params order) | Billions of parameters |
| **Align** | Teach helpful, safe, on-topic replies (SFT / preference) | Light assistant-style text | Large instruction + preference training |

```text
Quality  ≈  f(data, size, align)
```

UI knobs (`max_tokens`, `temp`) only **shape** the output.  
They do **not** replace data, size, or alignment.

---

## max_tokens: AERIS vs top models

| Setting | AERIS_64 (recommended) | Top models (typical) |
|---------|------------------------|----------------------|
| **max_tokens** | **Default 48; API/UI allow up to 8192** (1k–8k like top apps) | **Much higher OK** — often 1k–8k+ per reply |
| Why | Small model; long outputs dump training text / garbage | Large model stays coherent for long answers |
| **temperature** | **0.2–0.4** | Often 0.2–0.7 depending on task |

**“Usually much higher OK”** applies to **frontier-scale** models after full data + size + align — **not** to AERIS_64 with 255 tokens.

### Practical AERIS UI defaults

```text
max tokens:  48
temp:        0.2
```

If replies dump many Q&A lines: lower max_tokens first, then improve data.

---

## How to grow AERIS quality (order)

1. **Data** — more clean English (then other domains), short clear User/Assistant pairs  
2. **Size** — later `--preset small` / `medium` when CPU/GPU allows  
3. **Align** — more helpful Q&A examples; refuse / “I don’t know” examples  
4. Only then consider longer max_tokens  

---

## Related

- `docs/RUN_AND_TRAIN.md` — train commands  
- `docs/FRONTIER_TRAINING_ROADMAP.md` — Day 1…N plan  
- `docs/MODEL_CARD.md` — limits  
- `docs/PRETRAINING_OVERVIEW.md` — next-token + mixtures  


## AERIS max_tokens ceiling (updated)

- **Allowed:** `1` … **`8192`** (same order as top-model UIs: 1k–8k)
- **Default / recommended for quality:** **48**
- Raising the limit does **not** add intelligence — long replies still need **data + size + align**
- If you set 2000+ on tiny AERIS, expect slower replies and more dump/garbage until the model is larger and better trained

## Auto length (adaptive max_tokens)

Server flag **`auto_length: true`** (default):

| User input | Approx token budget |
|------------|---------------------|
| `hello` / `thanks` / short | ~40 (short answer) |
| Normal question | ~64–256 |
| `summarize book` / long ask | ~1024–2048 (up to your **cap**, max 8192) |

This only chooses **how long** to generate — **not** the answer text (no hardcode).  
Long useful summaries still need the book text in **training data** + enough model size.
