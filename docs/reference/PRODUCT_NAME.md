# Product vs architecture

| Field | Value |
|-------|--------|
| **Product / model name** | **AERIS_64** |
| **Architecture** | **NeuroField v2** (two-speed memory) |

- API `model` id: `AERIS_64`
- Checkpoint folder can stay `docs/neurofield_company` or rename to `docs/AERIS_64`
- Python package imports remain `neurofield.*` (code path)
- UI title: AERIS_64

Train:
```bash
PYTHONPATH=src python scripts/train_company.py --preset tiny --steps 800 --out docs/AERIS_64
CHECKPOINT=docs/AERIS_64 PYTHONPATH=src uvicorn "neurofield.serving.server:create_app" --factory --host 0.0.0.0 --port 8000
```
