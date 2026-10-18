#!/usr/bin/env bash
# First AERIS-42M train on current pretrain_42m corpus (~2.4M words / 24 lakh).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/packages${PYTHONPATH:+:$PYTHONPATH}"

echo "[stats]"
python scripts/data/stats_pretrain_42m.py || true

DEVICE="${NEUROFIELD_DEVICE:-auto}"
STEPS="${STEPS:-8000}"
OUT="${OUT:-checkpoints/AERIS_42m_2p4M}"
EXTRA=()
if python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  EXTRA+=(--bf16)
  DEVICE=cuda
  echo "[gpu] cuda + bf16"
else
  DEVICE=cpu
  STEPS="${SMOKE_STEPS:-200}"
  echo "[cpu] smoke steps=$STEPS"
fi

python scripts/train/train_company.py \
  --preset aeris_42m \
  --tokenizer bpe --bpe-vocab 32000 \
  --data data/pretrain_42m/en data/pretrain_42m/hi data/pretrain_42m/bn \
  --steps "$STEPS" \
  --batch-size "${BATCH:-8}" \
  --seq-len "${SEQ:-512}" \
  --grad-accum "${ACCUM:-4}" \
  --lr 3e-4 --warmup 300 --cosine --dropout 0.05 \
  --device "$DEVICE" \
  --out "$OUT" \
  --product AERIS_42m_2p4M \
  "${EXTRA[@]}"

echo "[done] $OUT"
