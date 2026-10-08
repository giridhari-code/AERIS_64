# NeuroField / AERIS — v2.2

**Author:** Giridhari · **Contact:** giriisdev@gmail.com  
**License:** personal / non-commercial terms — see [LICENSE](LICENSE), [NOTICE](NOTICE), [DATA_POLICY.md](DATA_POLICY.md).  
Data removal or legal concerns about files in this repo → **giriisdev@gmail.com**.

Research architecture (**NeuroField two-speed memory**) + inference API + training toolkit,
now with a **1.0B-parameter configuration** (`configs/aeris_1b.yaml`).

> **Read this first**
> * The 1B model is a **config + training recipe**. **No trained 1B weights are included** —
>   training one needs real data and a large GPU (see [docs/TRAIN_1B.md](docs/TRAIN_1B.md)).
> * The checkpoints in `docs/` (`AERIS_64`, `neurofield_*`) are **~60k-parameter demos** trained on
>   ~10 KB of hand-written text. They memorise that text; they are not assistants.
> * Architecture quality at 1B is **unproven**. Benchmark against a standard Transformer first.

## Quick start

```bash
pip install -e ".[serve,dev]"            # + ".[data]" for tokenizers
make test

# serve a checkpoint (folder with model.safetensors [+ tokenizer.json])
NEUROFIELD_CHECKPOINT=docs/neurofield_india PYTHONPATH=src \
  uvicorn "neurofield.serving.server:create_app" --factory --port 8000
```

Open http://127.0.0.1:8000/

## 1B model

```bash
python scripts/count_params.py configs/aeris_1b.yaml        # 1,004,511,364 params, no torch needed
python scripts/prepare_data.py --input "data/raw/*.txt" --out-dir data/tokens --vocab-size 32000
python -m neurofield.cli --config configs/aeris_1b.yaml --tokenizer-json data/tokens/tokenizer.json
```

## Docs

| Doc | Content |
|-----|---------|
| [docs/TRAIN_1B.md](docs/TRAIN_1B.md) | 1B recipe, memory, data, caveats |
| [docs/AUDIT_V2_2.md](docs/AUDIT_V2_2.md) | Audit findings, what was fixed, what is still open |
| [docs/RUN_AND_TRAIN.md](docs/RUN_AND_TRAIN.md) | Install, run, train (small models) |
| [docs/API_REFERENCE.md](docs/API_REFERENCE.md) | HTTP API |
| [docs/MODEL_CARD.md](docs/MODEL_CARD.md) | Model card |
| [data/README.md](data/README.md) | What the bundled data is (and is not) |

## Structure

```
├── configs/                  # YAML configs
├── src/neurofield/
│   ├── model.py              # Core model (Appendix A)
│   ├── modules/              # dendrite, field, memory, router...
│   ├── data/                 # datasets
│   ├── training/             # trainer
│   ├── safety/               # audit + anomaly monitor
│   ├── serving/              # FastAPI inference server
│   └── utils/                # structured logging
├── deploy/
│   ├── docker/               # Dockerfile + compose + entrypoint
│   ├── k8s/                  # Deployment, Service, PDB
│   └── systemd/              # (optional unit files)
├── scripts/
├── tests/
├── Makefile
├── pyproject.toml
└── .env.example
```

## Install

```bash
pip install -e ".[serve,dev]"
```

## Training

```bash
neurofield-train \
  --config configs/default.yaml \
  --train-data /data/train.bin \
  --val-data /data/val.bin \
  --output-dir runs/exp1 \
  --device cuda
```

## Inference Server (Production)

```bash
# Local
export NEUROFIELD_CHECKPOINT=runs/exp1/checkpoint_best.pt
# correct: factory + env (NOT create_app(checkpoint=...) as a string)
NEUROFIELD_CHECKPOINT="$NEUROFIELD_CHECKPOINT" PYTHONPATH=src \
  uvicorn "neurofield.serving.server:create_app" --factory --host 0.0.0.0 --port 8000
  --host 0.0.0.0 --port 8000

# Docker
cd deploy/docker
docker compose up --build
```

### API

| Endpoint            | Method | Purpose                    |
|---------------------|--------|----------------------------|
| `/healthz`          | GET    | Liveness probe             |
| `/readyz`           | GET    | Readiness (model loaded)   |
| `/metrics`          | GET    | Prometheus-style metrics   |
| `/v1/completions`   | POST   | Generate tokens            |
| `/v1/reset`         | POST   | Reset session state        |

Example request:

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"input_ids":[1,2,3,4], "max_new_tokens":32, "temperature":0.8}'
```

## Kubernetes

```bash
kubectl apply -f deploy/k8s/deployment.yaml
```

- Non-root (uid 10001)
- Liveness + Readiness probes
- GPU resource requests
- PodDisruptionBudget
- Prometheus annotations

## Safety (built-in)

- Per-session state isolation
- Write / memory norm caps
- Audit log of gates, norms, entropy
- Online z-score anomaly detector
- Optional freeze-on-anomaly

## Makefile targets

```
make install-dev   # install with serve + test deps
make test          # run tests
make docker-build  # build image
make docker-run    # compose up
make lint          # ruff
```

## License

MIT

## Run & Train

See **[docs/RUN_AND_TRAIN.md](docs/RUN_AND_TRAIN.md)** for full install, serve, and training steps.


## Mini RL

See [docs/REINFORCEMENT_LEARNING.md](docs/REINFORCEMENT_LEARNING.md) and `scripts/train_rl.py`.


## Scale & optimize

See [docs/SCALE_AND_OPTIMIZE.md](docs/SCALE_AND_OPTIMIZE.md).


## Multi-machine / Colab training

See [docs/MULTI_MACHINE_TRAINING.md](docs/MULTI_MACHINE_TRAINING.md) — one official checkpoint, resume chain, no silent weight forks.


## Company-style training process

Full stage map + 7-day plan: [docs/TRAINING_PROCESS_COMPANY_STYLE.md](docs/TRAINING_PROCESS_COMPANY_STYLE.md).
