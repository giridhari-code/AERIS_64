# model.safetensors only — no .pt

## Save
Always: `model.safetensors`  
Never: `model.pt`

```bash
pip install safetensors
PYTHONPATH=src python scripts/train_company.py --data data/day1.txt --out docs/AERIS_main
```

## Load
`load_checkpoint` / server accept **only** `model.safetensors`.  
`.pt` → error (no fallback).

If you still have an old `.pt`, convert once then delete the pickle:

```bash
PYTHONPATH=src python -c "
from neurofield.checkpoint.io import convert_pt_to_safetensors
convert_pt_to_safetensors('old.pt', 'docs/AERIS_main/model.safetensors')
"
```

(Only if `convert_pt_to_safetensors` exists and you trust the file source.)
