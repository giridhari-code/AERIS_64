# Changelog

## Unreleased — per-user memory, parallel training, paper v3

### Per-user memory (paper v3 Section 11, ring variant)
- `serving/user_memory.py`: lossless per-chat transcripts + persisted slow ring (safetensors, atomic), per API key / `user_id`.
  Fast memory is never persisted. Chat rehydrates its window from the transcript; re-fed history is replayed with
  `write_slow=False` so the ring is not flooded; ring overflow now **compacts** (count-weighted fold) instead of overwriting
  (`slow_compact`, default on; `filled` holds counts, `filled.sum()` = total writes). Norm cap, audit log, anomaly gating,
  `GET/DELETE /v1/memory`, `GET /v1/memory/transcript`, `remember=false`.
- `NeuroField.new_state(..., slow=)`, `export_slow_state()`, `forward(write_slow=)`.

### Parallel training
- Token-parallel forward (embedding, dendrite window, readout + head batched over T): exact same outputs, 1.3-1.5x faster
  fwd+bwd on CPU in `scripts/bench/bench_parallel_ops.py`.
- DDP (`torchrun`): rank-sharded data, no-sync accumulation, rank-consistent replay with all-reduced slow update, rank-0 I/O,
  fused AdamW, TF32. Tested with a 2-process gloo run (ranks stay bit-close; test fails if the replay all-reduce is removed).
- `lambda_err_warmup_steps` (paper v3 proposal, default off), embedding-norm logging with collapse warning.

### Paper
- v3 report added at `docs/paper/AESC_v2_Technical_Report_v3.pdf` (README now links it).

## Earlier — security + train/serve consistency patch

### Security
- Sandbox: commands are parsed with `shlex` and run **without a shell** (`echo hi; id` is now literal text);
  `bash`/`sh` removed from the allow-list; absolute/relative binary paths refused.
- Sandbox: network namespace (`unshare`) when available, `require_isolation` option, new RLIMIT_FSIZE/NOFILE/NPROC,
  `start_new_session`, `python3 -I -B`.
- Sandbox: `write_file` containment uses `is_relative_to` (sibling-dir escape fixed); session dir names include a
  digest of the full id (64-char truncation collision fixed); work dirs are deleted on eviction/drop.
- Server: `/v1/tools/shell|python|audit|reset` require an API key (503 if none configured;
  `NEUROFIELD_TOOLS_ALLOW_OPEN=1` for local dev), are rate-limited, and sandboxes are scoped per API key.
- Server: `/v1/tools/shell|python|reset` always returned 422 (request models were defined inside `create_app()`
  and FastAPI could not resolve them); models are now module-level. Auth hardening above matters because of this.
- Server: default tool `session_id` is now `default-session` (the old `default` failed the 8-char id regex);
  `web_search.max_results` clamped.

### Model
- Slow ring is written every `segment_len` tokens via a carried `state["seg_pos"]`; 1-token decode steps no longer
  write a slot each (train/serve mismatch). Slot weight = per-sequence relative surprise (no batch-mean norm).
  Behaviour change: re-validate / re-train checkpoints trained with the old rule.

### Model (paper Appendix B checks)
- Metacognition's module-level running baselines (`track_global`) now apply in **training only**. In `eval()` the same
  input used to give different logits on a second call (max diff ~4.9 on a tiny model) because `bar_S`/entropy baselines
  carried over between calls and between sessions; only the server disabled this manually.
- New `tests/test_paper_checklist.py`: leakage/causality test (checklist item 8 / Proposition 7 had no test; it passes:
  future tokens change past logits by exactly 0), targets-don't-affect-logits, and per-call isolation.

### Repo
- Original v2 report PDF archived unmodified in `docs/paper/archive/` (contains H1; see its README).
- Added `SECURITY.md`, `.dockerignore`, a demo-checkpoint smoke test (`tests/test_example_checkpoint.py`) and an advisory `pip-audit` CI job.
- Technical report PDF regenerated from `scripts/docs/make_tech_report_pdf.py` (H1 removed; edit text there, not in the PDF).
- Removed the H1 ("10M AESC ≈ 100M Transformer") hypothesis from README, technical report (.md), research note and audit docs. The PDF copy of the report still contains it until regenerated.
- `normalize_prompt` now implements the `namste`/`vanakam` fixes that `test_normalize_hindlish` already expected.
- `pyproject` license metadata now matches `LICENSE` (was "MIT"); Makefile/k8s versions and default checkpoint path;
  `convert_pt_to_safetensors` works again; calculator no longer touches `ast.Num` (removed in Python 3.14);
  place-name regex no longer matches lowercase words.

## 2.3.0 — 2026-10-09

### Repository
- Professional layout: `docs/{paper,guides,reference,audits,archive}`, `scripts/{train,eval,data,research}`, `checkpoints/examples`
- Incomplete weight-less folders moved out of `docs/` into `checkpoints/examples/`
- Clean root README and docs indexes

### Model / research
- Fast/slow memory production fixes (decay, W_q/W_k, zero slow init, gate bias)
- Brain-inspired metacognition (monitor + control)
- Tool sandbox (`/v1/tools/*`)
- Contribution ablation script
- AESC v2.3 technical report under `docs/paper/`

## 2.2.0

- Prior NeuroField / AERIS public tree (audits, 1B config recipe, serving stack)
