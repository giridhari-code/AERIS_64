# Checkpoints

**Weights are not stored in git.** Train locally or copy `model.safetensors` into a run folder.

## Layout

```text
checkpoints/
  README.md                 # this file
  examples/                 # config + tokenizer side-cars only (no weights)
  local_run/                # your train output (gitignored pattern via *.safetensors)
```

## Serve

```bash
# After training into checkpoints/local_run (must include model.safetensors)
export NEUROFIELD_CHECKPOINT=checkpoints/local_run
export PYTHONPATH=src
uvicorn "neurofield.serving.server:create_app" --factory --port 8000
```

## Why examples have no weights

`.gitignore` excludes `*.safetensors`. The folders under `examples/` keep tokenizer/config metadata for reference only.
