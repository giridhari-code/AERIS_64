#!/usr/bin/env bash
# Decent assistant training: preset large (d_model=512) + BPE + 1M-token-scale data.
# GPU strongly recommended. Without CUDA this is a smoke path only.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/packages${PYTHONPATH:+:$PYTHONPATH}"

DEVICE="${NEUROFIELD_DEVICE:-auto}"
OUT="${OUT:-checkpoints/AERIS_decent}"
STEPS="${STEPS:-20000}"

mkdir -p data/decent
if [[ -z "$(find data/decent -name '*.txt' -type f 2>/dev/null | head -1)" ]]; then
  echo "[decent] no data/decent/*.txt — generating seed (not enough for 1M tokens)"
  python scripts/data/make_synthetic_100k.py \
    --out data/decent/synthetic_seed.txt --target-chars 200000
fi

mapfile -t DATA < <(find data/decent -name '*.txt' -type f | sort)
CHARS=$(cat "${DATA[@]}" 2>/dev/null | wc -c | tr -d ' ')
echo "[decent] ${#DATA[@]} files, ~${CHARS} characters (aim for several million chars ≈ 1M+ tokens)"

EXTRA=()
if [[ "${DEVICE}" == "cuda" ]] || [[ "${DEVICE}" == "auto" ]]; then
  if python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    EXTRA+=( --bf16 )
    echo "[decent] CUDA available — bf16 on"
  else
    echo "[decent] WARNING: no GPU — training will be very slow; reduce STEPS for smoke"
    STEPS="${SMOKE_STEPS:-200}"
    DEVICE=cpu
  fi
fi

python scripts/train/train_company.py \
  --preset large \
  --tokenizer bpe \
  --bpe-vocab 4000 \
  --data "${DATA[@]}" \
  --steps "${STEPS}" \
  --batch-size "${BATCH:-8}" \
  --seq-len 512 \
  --grad-accum "${ACCUM:-4}" \
  --lr 3e-4 \
  --warmup 500 \
  --cosine \
  --dropout 0.1 \
  --label-smoothing 0.1 \
  --device "${DEVICE}" \
  --out "${OUT}" \
  --product AERIS_decent \
  --identity-marker AERIS \
  "${EXTRA[@]}"

echo "[decent] done → ${OUT}"
echo "Serve: CHECKPOINT=${OUT} make backend"
