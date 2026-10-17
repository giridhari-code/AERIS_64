#!/usr/bin/env bash
# Train SENTINEL on the ~1M-token Dataset Builder corpus (Colab / local GPU).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${PYTHONPATH:-}:packages"

DATA="${DATA:-artifacts/dataset_1m/aeris64_1m_train.jsonl}"
OUT="${OUT:-checkpoints/aeris64_1m_ft}"
STEPS="${STEPS:-1200}"
PRESET="${PRESET:-small}"
DEVICE="${DEVICE:-auto}"
RESUME="${RESUME:-}"

if [[ ! -f "$DATA" ]]; then
  echo "Missing $DATA — build first:"
  echo "  PYTHONPATH=packages python3 scripts/data/build_1m_corpus.py"
  exit 1
fi

ARGS=(
  --data "$DATA"
  --out "$OUT"
  --steps "$STEPS"
  --preset "$PRESET"
  --device "$DEVICE"
  --grad-accum 4
  --warmup 80
  --cosine
  --lr 5e-4
  --product AERIS_64
)
if [[ -n "$RESUME" ]]; then
  ARGS+=(--resume "$RESUME")
fi

echo "1M train → python scripts/train/train_company.py ${ARGS[*]}"
exec python scripts/train/train_company.py "${ARGS[@]}"
