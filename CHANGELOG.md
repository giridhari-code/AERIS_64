# Changelog

## 2.2.0
* **1B configuration**: `configs/aeris_1b.yaml` / preset `aeris_1b` = 1,004,511,364 parameters (no trained weights).
* Trainer: grad accumulation, bf16 autocast, resume, atomic checkpoints, safetensors export, no-decay groups,
  `truncate_write_window` honoured, non-finite-grad skip.
* Data: `scripts/prepare_data.py` (dedupe, held-out split, HF tokenizer, uint16 bins), HF tokenizer wrapper.
* Checkpoint: shape-based config inference, strict load, legacy `slow_mem.W_s` remap, safe `.pt` loading.
* Server rewritten (thread-safe, session TTL/owner, fresh-state chat, working reset, real streaming path, `--factory` deploy).
* Eval no longer measures validation on the training corpus.
* Model forward: per-token GPU syncs removed (same math).
* Tests: server end-to-end, shape inference, leak-free split, param formula; fixed vacuous assertion; CI installs serve extras.
* See `docs/AUDIT_V2_2.md` for the full list and the items that remain open.
