# Missing items — what was added

| Previously missing | What we added |
|--------------------|---------------|
| Bigger capacity | `configs/presets.yaml` — **tiny / small / medium**; `model_preset()` |
| Longer context | preset `max_seq_len` 128→512; train `--seq-len` / preset default |
| BPE path | already in train `--tokenizer bpe` + server tokenizer.json |
| Cost vs Transformer measure | `scripts/benchmark_cost.py` micro-benchmark |
| Multi-domain data | train accepts multiple `--data` files (company + code + india + skills) |
| Train with preset | `train_company.py --preset small` |

## Still not “magic complete”

| Item | Why not fully addable in-repo |
|------|-------------------------------|
| Frontier CUDA kernels | Needs dedicated GPU eng |
| Proven 10× cheaper than Transformer | Needs your benchmarks on target hardware |
| Perfect multi-domain fluency | Needs **more data + more steps**, not one module |

## Commands

```bash
# larger capacity (GPU better for medium)
PYTHONPATH=src python scripts/train_company.py --preset small --steps 800 --out docs/neurofield_company

# long-ish sequences
PYTHONPATH=src python scripts/train_company.py --preset medium --seq-len 128 --steps 500

# cost micro-benchmark vs mini Transformer
PYTHONPATH=src python scripts/benchmark_cost.py --preset tiny --seq 64 --batch 8

# BPE
PYTHONPATH=src python scripts/train_company.py --tokenizer bpe --bpe-vocab 400 --preset small --steps 800
```
