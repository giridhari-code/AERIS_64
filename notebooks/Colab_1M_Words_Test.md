# AERIS_64 — Colab 1M words pipeline test

**Runtime → GPU (T4)**

## 1) Clone + install

```python
!nvidia-smi -L || true
import os
os.chdir('/content')
!rm -rf AERIS_64
!git clone --depth 1 https://github.com/giridhari-code/AERIS_64.git
%cd AERIS_64
!pip -q install -e ".[dev]" tokenizers
```

## 2) One-shot: 1M words → tokenize → train (500 steps)

```python
import os
os.environ['PYTHONPATH'] = '/content/AERIS_64/packages'
!python scripts/train/colab_1m_words_test.py --words 1000000 --max-steps 500
```

### Faster smoke (optional)

```python
!python scripts/train/colab_1m_words_test.py --words 200000 --max-steps 50
```

## 3) What success looks like

- `data/raw/colab_1m.txt` ≈ 1M words
- `data/tokens_1m/train.bin` + `tokenizer.json`
- `runs/colab_1m_test/` checkpoints
- Train log: finite `loss`, `ce`, no NaN / skip-update spam

## Notes

- Config: `configs/colab_1m_test.yaml` (small model, Colab-safe)
- Corpus is **synthetic** (pipeline test, not quality pretrain)
- For real text: put `.txt` under `data/raw/` and run `scripts/data/prepare_data.py` yourself
