# model.safetensors only — no .pt

## Save
Always: `model.safetensors`  
Never: `model.pt`

```bash
pip install safetensors
PYTHONPATH=src python scripts/train/train_company.py --data data/samples/company_corpus.txt --out checkpoints/local_run
```

## Load
`load_checkpoint` / server accept **only** `model.safetensors`.  
`.pt` → error (no fallback).

If you still have an old `.pt`, convert once then delete the pickle:

```bash
PYTHONPATH=src python -c "
from neurofield.checkpoint.io import convert_pt_to_safetensors
convert_pt_to_safetensors('old.pt', 'checkpoints/local_run')
"
```

(Uses `torch.load(weights_only=True)`; set `NEUROFIELD_ALLOW_PICKLE=1` only for a file you made yourself.)
