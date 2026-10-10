# NeuroField / AERIS

**Two-speed memory architecture** for recurrent language models  
Research code · production-oriented layout · not a finished frontier product

| | |
|---|---|
| **Author** | Giridhari Karmakar · giriisdev@gmail.com |
| **Paper** | [docs/paper/AESC_v2_3_Technical_Report.pdf](docs/paper/AESC_v2_3_Technical_Report.pdf) |
| **License** | See [LICENSE](LICENSE), [NOTICE](NOTICE), [DATA_POLICY.md](DATA_POLICY.md) |

---

## What this is

NeuroField (AESC) is a recurrent LM with:

- k-step neural field  
- predictive dendrite (delayed error)  
- top-k skill routing  
- **fast memory** (surprise-gated delta rule)  
- **slow memory** (replay consolidation)  
- metacognition (monitor → control)  
- optional tool **sandbox**

The small checkpoints under `checkpoints/examples/` are **side-car configs only** (weights are not in git). Train locally to serve a model.

---

## Repository layout

```text
AERIS_64/
├── src/neurofield/          # Library (model, modules, serve, train, sandbox)
├── configs/                 # YAML configs (default, presets, 1B recipe)
├── scripts/
│   ├── train/               # Training entrypoints
│   ├── eval/                # Evaluation
│   ├── data/                # Data prep
│   └── research/            # Ablations, param counts, probes
├── tests/                   # pytest
├── data/
│   ├── samples/             # Tiny demo corpora
│   └── align/               # Alignment text
├── checkpoints/
│   └── examples/            # Config/tokenizer side-cars (no .safetensors in git)
├── docs/
│   ├── paper/               # Technical reports
│   ├── guides/              # How to train / scale
│   ├── reference/           # API, model card, tokens
│   ├── audits/              # Engineering audits & patch notes
│   └── archive/             # Historical notes
├── deploy/                  # Docker, k8s, systemd
├── artifacts/               # Local outputs (gitignored content)
└── pyproject.toml
```

---

## Quick start

```bash
git clone https://github.com/giridhari-code/AERIS_64.git
cd AERIS_64
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[serve,dev]"
```

### Tests

```bash
make test
# or: PYTHONPATH=src pytest tests/ -q
```

### Train a small local checkpoint (then serve)

```bash
export PYTHONPATH=src
python scripts/train/train_company.py \
  --data data/samples/company_corpus.txt data/samples/india_multilang.txt \
  --tokenizer char --steps 500 --d-model 64 \
  --out checkpoints/local_run

# Serve (folder must contain model.safetensors)
export NEUROFIELD_CHECKPOINT=checkpoints/local_run
uvicorn "neurofield.serving.server:create_app" --factory --host 0.0.0.0 --port 8000
```

Open http://127.0.0.1:8000/

### Research ablation (associative recall)

```bash
PYTHONPATH=src python scripts/research/contribution_ablation.py \
  --steps 800 --seeds 0 --batch 8 --d-model 64 \
  --ablations full,no_fast_mem,no_slow_mem,no_neuromod,no_metacog,skills_only \
  --device cpu --out artifacts/contrib.json
```

---

## Important

- **Weights are not in git** (`*.safetensors` ignored) with ONE exception: the tiny demo `checkpoints/examples/AERIS_main/model.safetensors` (~2.7 MB). See [docs/reference/WEIGHTS_NOT_IN_GIT.md](docs/reference/WEIGHTS_NOT_IN_GIT.md).  
- The other folders under `checkpoints/examples/` have config/vocab only — serving them will fail until you train or copy weights in.  
- **Sandbox tools (`/v1/tools/shell|python`) need `NEUROFIELD_API_KEYS`** (they answer 503 otherwise) and are defence-in-depth only: run the server in a container without secrets or network. See [docs/audits/PATCH_NOTES_SECURITY_2026-10.md](docs/audits/PATCH_NOTES_SECURITY_2026-10.md).  
- 1B config: `configs/aeris_1b.yaml` — recipe only, no pretrained 1B weights.  
- H1 (10M AESC ≈ 100M Transformer on memory tasks) is **untested**. See the v2.3 paper.

---

## Docs index

| Path | Content |
|------|---------|
| [docs/paper/](docs/paper/) | AESC v2.3 technical report |
| [docs/guides/RUN_AND_TRAIN.md](docs/guides/RUN_AND_TRAIN.md) | Install, run, train |
| [docs/guides/TRAIN_1B.md](docs/guides/TRAIN_1B.md) | 1B recipe |
| [docs/reference/API_REFERENCE.md](docs/reference/API_REFERENCE.md) | HTTP API |
| [docs/audits/](docs/audits/) | Audits and patch notes |

---

## Citation

If you use this code or report:

> Karmakar, G. (2026). *AESC v2.3: Two-Speed Memory Architecture (Revised Technical Report)*. https://github.com/giridhari-code/AERIS_64


## Component reference

Har module ka order + kaam: [docs/COMPONENTS_FULL_GUIDE.md](docs/COMPONENTS_FULL_GUIDE.md).
