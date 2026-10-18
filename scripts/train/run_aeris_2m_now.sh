#!/usr/bin/env bash
# Train ~2.3M-param model on current ~2.4M-word corpus
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/packages${PYTHONPATH:+:$PYTHONPATH}"
python scripts/data/stats_pretrain_42m.py || true

DEVICE=cpu
EXTRA=()
STEPS="${STEPS:-6000}"
if python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  DEVICE=cuda; EXTRA+=(--bf16); echo "[gpu]"
else
  STEPS="${SMOKE_STEPS:-300}"; echo "[cpu] steps=$STEPS"
fi

python scripts/train/train_company.py \
  --preset aeris_2m --tokenizer bpe --bpe-vocab 8000 \
  --data data/pretrain_42m/en data/pretrain_42m/hi data/pretrain_42m/bn \
  --steps "$STEPS" --batch-size "${BATCH:-16}" --seq-len 256 --grad-accum 2 \
  --lr 3e-3 --warmup 200 --cosine --dropout 0.05 \
  --device "$DEVICE" --out "${OUT:-checkpoints/AERIS_2m}" --product AERIS_2m \
  "${EXTRA[@]}"
