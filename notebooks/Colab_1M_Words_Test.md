# AERIS_64 — Colab 1M words test (existing pipeline only)

Uses the **normal** train path only:

```text
raw .txt  →  scripts/data/prepare_data.py  →  train.bin/val.bin
          →  python -m sentinel.cli train --config configs/colab.yaml
```

No extra train scripts.

## 1) Setup

```python
!nvidia-smi -L || true
import os
os.chdir('/content')
!rm -rf AERIS_64
!git clone --depth 1 https://github.com/giridhari-code/AERIS_64.git
%cd AERIS_64
!pip -q install -e ".[dev]" tokenizers
import os
os.environ['PYTHONPATH'] = '/content/AERIS_64/packages'
```

## 2) ~1M words corpus (your text or quick synthetic)

**Option A — your data**

```python
# put .txt files under data/raw/
!mkdir -p data/raw
# upload or download your corpus into data/raw/
```

**Option B — quick synthetic ~1M words for pipeline test**

```python
from pathlib import Path
p = Path('data/raw/colab_1m.txt')
p.parent.mkdir(parents=True, exist_ok=True)
words, n = 0, 0
with p.open('w') as f:
    while words < 1_000_000:
        line = (
            f"AERIS doc {n} memory field skills predict error. "
            f"unique_id={n} hex={n:08x} value={n%97}.\n\n"
        )
        f.write(line)
        words += len(line.split())
        n += 1
print('words', words)
```

## 3) Prepare tokens (official script)

```python
!python scripts/data/prepare_data.py \
  --input "data/raw/*.txt" \
  --out-dir data/tokens \
  --vocab-size 8000 \
  --val-frac 0.01 \
  --chunk-chars 2000
```

## 4) Train (official CLI + configs/colab.yaml)

```python
!python -m sentinel.cli train \
  --config configs/colab.yaml \
  --data-dir data/tokens \
  --tokenizer-json data/tokens/tokenizer.json \
  --max-steps 500
```

Checkpoints: `runs/colab/`

## Success

- Finite loss / ce in logs  
- `runs/colab/checkpoint_*.pt` (or latest)  
- No NaN skip spam  
