# Changelog

## Unreleased — security + train/serve consistency patch

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

### Repo
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
