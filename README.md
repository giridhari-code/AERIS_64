# SENTINEL

*Think Smarter. Do More*

**Two-speed memory architecture** (SENTINEL · formerly SENTINEL / AERIS) for recurrent language models  
Research code · production-oriented layout · not a finished frontier product

| | |
|---|---|
| **Author** | Giridhari Karmakar · giriisdev@gmail.com |
| **Full research paper (publish draft)** | [docs/paper/AESC_SENTINEL_Full_Research_Paper_Giridhari_Karmakar.pdf](docs/paper/AESC_SENTINEL_Full_Research_Paper_Giridhari_Karmakar.pdf)
| **Paper (final handoff)** | [docs/paper/AESC_SENTINEL_Paper_v2_Giridhari_Karmakar.pdf](docs/paper/AESC_SENTINEL_Paper_v2_Giridhari_Karmakar.pdf) · prior: [v3](docs/paper/AESC_v2_Technical_Report_v3.pdf) · code-level report: [v2.3](docs/paper/AESC_v2_3_Technical_Report.pdf) |
| **License** | See [LICENSE](LICENSE), [NOTICE](NOTICE), [DATA_POLICY.md](DATA_POLICY.md) |

---

**Handoff:** [docs/PROJECT_HANDOFF.md](docs/PROJECT_HANDOFF.md) · **Full paper:** [docs/paper/AESC_SENTINEL_Full_Research_Paper_Giridhari_Karmakar.pdf](docs/paper/AESC_SENTINEL_Full_Research_Paper_Giridhari_Karmakar.pdf)

## What this is

SENTINEL is a recurrent LM with two-speed memory (SENTINEL lineage) with:

- k-step neural field  
- predictive dendrite (delayed error)  
- top-k skill routing  
- **fast memory** (surprise-gated delta rule)  
- **slow memory** (replay consolidation)  
- metacognition (monitor → control)  
- optional tool **sandbox**

The small checkpoints under `checkpoints/examples/` are **side-car configs only** (weights are not in git). Train locally to serve a model.

### Dual stack

| Layer | Stack | Role |
|-------|--------|------|
| **Model** | Python | Training, neural field, fast/slow memory, inference (`packages/sentinel`) |
| **Frontend** | TypeScript + Bun | Chat UI only (`frontend`) — proxies to Python API |

```bash
# Python model server
uvicorn "sentinel.api.app:create_app" --factory --port 8000

# Bun edge (proxies chat → Python, exposes /v1/git/*)
cd frontend && bun install && bun run dev   # :3000
```

See [frontend/README.md](frontend/README.md).

---

## Repository layout

```text
AERIS_64/
├── packages/sentinel/     # Model library (train, modules, API, tools)
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
├── frontend/           # TypeScript + Bun UI only (proxies to :8000)
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
# or: PYTHONPATH=packages pytest tests/ -q
```

### Train a small local checkpoint (then serve)

```bash
export PYTHONPATH=packages
python scripts/train/train_company.py \
  --data data/samples/company_corpus.txt data/samples/india_multilang.txt \
  --tokenizer char --steps 500 --d-model 64 \
  --out checkpoints/local_run

# Serve (folder must contain model.safetensors)
export NEUROFIELD_CHECKPOINT=checkpoints/local_run
uvicorn "sentinel.api.app:create_app" --factory --host 0.0.0.0 --port 8000
```

Open http://127.0.0.1:8000/

### Research ablation (associative recall)

```bash
PYTHONPATH=packages python scripts/research/contribution_ablation.py \
  --steps 800 --seeds 0 --batch 8 --d-model 64 \
  --ablations full,no_fast_mem,no_slow_mem,no_neuromod,no_metacog,skills_only \
  --device cpu --out artifacts/contrib.json
```

---

## Per-user memory (fast = the model's own, slow = the user's)

* **Fast memory** (`M`, field state, dendrite history) is the model's working memory. It is never written to disk and starts
  from zero every turn.
* **Slow memory** is the user's. With `NEUROFIELD_MEMORY_DIR` + `NEUROFIELD_API_KEYS` set, `/v1/chat` (and `/v1/chat/stream`)
  with a `user_id`:
  * stores **every turn verbatim** (append-only `chats/<session>.jsonl`, rotated by size, never truncated) and rebuilds the chat
    window from it after a restart / TTL eviction;
  * persists the user's slow ring (`slow.safetensors`, shared by all that user's chat windows). When the ring is full the oldest
    slot is **folded into its neighbour (count-weighted)** instead of being erased, so old context is diluted, not dropped;
  * caps slot norms and writes an `audit.jsonl` line per consolidation; a turn flagged by the anomaly monitor is **not** persisted;
  * is isolated per API key and `user_id`; `GET /v1/memory`, `GET /v1/memory/transcript`, `DELETE /v1/memory?user_id=` let the
    user inspect and erase it. `remember=false` skips storing a turn.

  This is the ring-based variant of the paper's proposed per-user context memory `C_u` (v3, Section 11). Whether it improves
  answers has **not** been measured; the tests cover the mechanics (nothing lost, isolation, restart, deletion).

## Parallel / faster training

* **Token-parallel forward** (on by default, `NEUROFIELD_PARALLEL_OPS=0` falls back to the per-token loops): embedding, the dendrite
  window, the skill router, the neural-field step, the slow-memory read and the whole readout + vocabulary head are computed for
  all `T` tokens at once; only the genuinely sequential fast-memory delta-rule recurrence stays in a loop. Per-token gate/memory
  norms are only collected on logged steps (`collect_stats`). `tests/test_parallel_training.py` compares both paths;
  `python scripts/bench/bench_parallel_ops.py` measures the speed-up.
* **Data parallel (DDP)**: `torchrun --nproc_per_node=N -m sentinel.cli train ...` shards the data per rank, all-reduces
  gradients (only on the last micro-batch of an accumulation), and all-reduces the slow-memory replay so ranks cannot drift.
  Also: gradient accumulation, bf16, fused AdamW and TF32 on CUDA, optional `lambda_err_warmup_steps` (paper v3 proposal, off by
  default), and embedding-norm logging (Appendix B item 14).
* The recurrence itself is still sequential in time (the paper's stated limitation); this removes everything around it.

## Important

- **Weights are not in git** (`*.safetensors` ignored) with ONE exception: the tiny demo `checkpoints/examples/AERIS_main/model.safetensors` (~2.7 MB). See [docs/reference/WEIGHTS_NOT_IN_GIT.md](docs/reference/WEIGHTS_NOT_IN_GIT.md).  
- The other folders under `checkpoints/examples/` have config/vocab only — serving them will fail until you train or copy weights in.  
- **Sandbox tools (`/v1/tools/shell|python`) need `NEUROFIELD_API_KEYS`** (they answer 503 otherwise) and are defence-in-depth only: run the server in a container without secrets or network. See [docs/audits/PATCH_NOTES_SECURITY_2026-10.md](docs/audits/PATCH_NOTES_SECURITY_2026-10.md).  
- 1B config: `configs/aeris_1b.yaml` — recipe only, no pretrained 1B weights.  

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
