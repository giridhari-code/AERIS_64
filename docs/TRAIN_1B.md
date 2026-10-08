# Training the 1B configuration

**Status:** architecture + recipe + tooling. **No 1B weights are shipped.**
Nothing here has been run on a GPU by the author of this update (no GPU/network was available);
treat all commands as a starting point and expect to debug.

## Size (exact, verified against real checkpoint headers)

`python scripts/count_params.py configs/aeris_1b.yaml`

| Part | Params |
|---|---:|
| 16 skill MLPs (d→2d→d) | 822,255,616 |
| Tied embedding / head (32k × 3584) | 114,688,000 |
| Neural field (conv + W_m) | 39,015,424 |
| Predictor | 25,697,280 |
| fast/slow memory, P_f/P_s, router, misc | ~2.9 M |
| **Total** | **1,004,511,364** |

82% of the weights are the skill MLPs. Note `top_k=2` only gates the *weights*: the current router
still **computes all 16 skills for every token** (needed for the straight-through gradient), so
per-token compute is that of a dense 1B model, not a 2/16 MoE.

## Memory (estimates, not measurements)

fp32 weights 4.0 GB + grads 4.0 GB + AdamW 8.0 GB = **~16 GB static**, plus activations. The model runs a
Python loop over time, so activation memory grows with `batch × seq_len`. Start with
`batch_size 2–4, seq_len 256, bf16` and increase until you hit your GPU limit.
Single GPU only — there is no DDP/FSDP path yet.

## Speed — measure before you commit

The forward pass is **sequential in time** (one recurrent step per token). Throughput is limited by
kernel-launch overhead and memory bandwidth, not by FLOPs, and will be far below a same-size
Transformer. I did not measure it. Do this first:

```bash
python scripts/benchmark_cost.py          # compares against a Transformer baseline
python -m neurofield.cli --config configs/aeris_1b.yaml --max-steps 20   # read the tok/s log line
```

Then the plan is arithmetic: a common rule of thumb is ~20 training tokens per parameter (≈ 20 B tokens).
`days = 20e9 / (tok_per_s × 86400)`. If that number is not acceptable, shrink the model (`presets.yaml`)
rather than training an under-fed 1B.

## Data

The bundled `data/*.txt` is ~10 KB of demo text. **Do not train the 1B on it** — it will memorise it.
You need billions of tokens you have the right to use (licences!). Then:

```bash
pip install -e ".[data]"
python scripts/prepare_data.py --input "data/raw/**/*.txt" --out-dir data/tokens --vocab-size 32000
```

`prepare_data.py` drops exact duplicate chunks, holds out a validation split by hash, trains the tokenizer
on the train side only, and writes `train.bin`/`val.bin` (uint16). It does **not** do near-duplicate removal,
quality/language filtering or licence checks — do those before spending GPU weeks.

## Train / resume / evaluate

```bash
python -m neurofield.cli --config configs/aeris_1b.yaml --tokenizer-json data/tokens/tokenizer.json
python -m neurofield.cli --config configs/aeris_1b.yaml --resume          # continue from checkpoint_latest.pt
neurofield-eval --checkpoint runs/aeris_1b/best --data data/tokens/val.bin --device cuda
```

Outputs in `runs/aeris_1b/`: `checkpoint_latest.pt` (resume; contains optimizer), and `best/`, `final/`
safetensors folders (weights + config + tokenizer) that the server loads directly.

## Serve

```bash
NEUROFIELD_CHECKPOINT=runs/aeris_1b/final NEUROFIELD_DEVICE=cuda NEUROFIELD_DTYPE=bfloat16 \
  uvicorn "neurofield.serving.server:create_app" --factory --port 8000
```

One process, one model, requests serialised by a lock (no batching). Fine for demos / low traffic.

## Before believing any result

1. Held-out cross-entropy only (`val.bin`); never the training text.
2. Compare to a Transformer of equal parameters and equal tokens.
3. Check the loss curve for NaNs; the trainer skips non-finite steps and logs a warning.
