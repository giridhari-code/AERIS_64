# Experiments, baselines, data, chunkwise

## Data

- `data/samples/experiment_pretrain.txt` — short pretrain text for CE experiments
- `data/align/sft_experiment.txt` — User/Assistant SFT lines

## Experiments

```bash
PYTHONPATH=src python scripts/eval/run_experiments.py --device cpu --recall-steps 150 --text-steps 100
# report: docs/audits/experiment_report_sample.json
```

Includes associative-recall smoke + AESC CE + TinyTransformer baseline CE on the same file.

## Baseline

`src/neurofield/baselines/transformer_lm.py` — minimal causal Transformer for equal-data CE compare (not a full GPT stack).

## Chunkwise speed (reference form)

```bash
PYTHONPATH=src python scripts/train/train_company.py \
  --data data/samples/experiment_pretrain.txt \
  --preset tiny --steps 200 --chunk-size 16 \
  --out docs/runs/chunk_v1 --device cpu
```

`--chunk-size C` runs the **same** looped `forward` on successive chunks of length C, carrying `state` (fast M, hist, …). `C=0` means full sequence (default). This is the practical paper-style chunked train; a full DeltaNet kernel is still future work.

## Honesty

These harnesses produce **small-scale** numbers for tables and debugging. They do not replace large multi-seed benchmarks.
