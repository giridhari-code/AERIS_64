# Audit & update notes — v2.2

Static review + changes. **Torch/GPU were unavailable here: pure-python logic (param formula, shape inference,
leak-free split, checkpoint-key checks, config files, shell syntax) was executed; everything touching
torch/fastapi was only compiled, not run.** Run `make test` first.

## Verified by execution
* Param formula == exact tensor counts of 3 real checkpoints (63,340 / 60,508 / 36,064) → 1B config = 1,004,511,364.
* `neurofield_trained` and `neurofield_ddia_trained` store the slow memory as `slow_mem.W_s`, the model expects
  `slow_mem.M_s`. The old `strict=False` loader **silently ran them with a random slow memory.** Fixed (key remap + strict load).
* AERIS_64 / company checkpoints: all expected keys present.

## Fixed
| Area | Problem | Fix |
|---|---|---|
| Eval | `val_nll` measured on the last 20% of the *training* text; duplicates on both sides | held-out only, dedupe-before-split, `None` when no val set |
| Eval | `identity_hit` hardcoded `"neurofield"` | parameter `identity_marker` |
| Eval | `daily_eval` crashed (`load_checkpoint(map_location=)`) | rewritten on shared loader |
| Server | `uvicorn "…create_app(checkpoint=…)"` invalid | `--factory` + env vars (README, Docker, systemd) |
| Server | `NEUROFIELD_CHECKPOINT/CONFIG/DEVICE` ignored | read first, short names still work |
| Server | yaml dims (256/32000) vs checkpoint dims → crash; `{}` configs → guessed dims | dims inferred from weights |
| Server | `strict=False` hid missing weights; retry repeated same call | strict by default |
| Server | `/v1/reset` never cleared history | clears session + history |
| Server | history re-fed while recurrent state carried → context seen twice | chat starts from fresh state |
| Server | global `reset_meta`, shared `bar_S` baseline across sessions | `track_global=False` in server |
| Server | no lock, blocking `async def`, unbounded sessions/history/limiter | `def` endpoints, lock, TTL + max sessions, bounded limiter |
| Server | stream endpoint skipped history/stop/polish/errors | unified path |
| Server | O(n²) decode per token | tail-window decode |
| Server | safety z-threshold hardcoded 3.0 (config 4.0 ignored) | from config |
| Server | logging never configured (`LOG_JSON` unused) | wired |
| Auth | key in query string, non-constant-time compare, client-chosen session ids, any key could reset any session | header only, `hmac.compare_digest`, validated ids, owner binding |
| Model | 4–5 `.item()` GPU syncs per token | 1 sync/token (same math) |
| Trainer | `truncate_write_window` ignored; no accumulation/bf16/resume; non-atomic save | implemented |
| Loader | `torch.load(weights_only=False)` | safe by default, `NEUROFIELD_ALLOW_PICKLE=1` to override |
| Data | same-seed workers duplicate batches; `pin_memory` on CPU | fixed |
| Tests | vacuous normaliser assertion; no server tests; CI missing fastapi | fixed / added |
| Packaging | `static/index.html` not packaged; version drift 2.0.0/2.1.0; Docker hardcoded py3.11 path | fixed |

## Still open (not fixed — be aware)
* **Demo data is canned.** `company_corpus`, `assistant_style`, `align/*` are hand-written Q&A; models trained on them memorise. The
  "constitution" is plain text with no mechanism. Alignment here is **not** CAI/RLAIF.
* `skills_real.txt` / DDIA text: provenance and licence unknown; DDIA corpus is not in the repo.
* Reply post-processing heuristics (first-line cut, length regexes, typo list) remain; disable with `"postprocess": false`.
* Legacy pickle `.pt` files remain in `docs/` (now loaded safely or refused). Convert them to safetensors when you can.
* Safety `freeze_writes()` is global and irreversible per process if enabled.
* Router still computes all skills per token. Single GPU, no batching in the server, UI has no API-key field.
* 1B quality is unmeasured; GPU path is untested.
