# Changelog

## Unreleased — 0zcode: real interactive TUI

- **0zcode now runs a real terminal UI** (std only, no external crates):
  raw-mode prompt in a rounded input box, readline-style editing
  (`ctrl+a/e/w/u/k`, `alt+b/f`), persistent input history, `/` command menu
  with Tab completion, bracketed paste, resize-aware in-place redraw,
  Devanagari/CJK-aware cursor, and terminal state restored on exit/panic.
- **Tool output as blocks**: one `● tool(query)` block per tool with a `⎿`
  result tree, long output collapsed (`ctrl+o` / `/expand`), tool failures
  shown as error blocks instead of raw frames.
- **Live spinner** while network tools run; `esc` / `ctrl+c` interrupts and
  type-ahead is preserved.
- **Plain mode** for pipes / `TERM=dumb` / `ZEROZ_PLAIN=1`. Fixes the old REPL
  never exiting at EOF when stdin was piped.
- **Company hardening**: AERIS-64 model label default, expanded code-policy
  markers, clearer refusals, optional `ZEROZ_AERIS_URL` LLM fallback.
- Module split: `term` (raw mode, key decoder, widths), `editor`, `tui`
  (pure renderers), `repl` (session loop). Tests 25 → 66. Removed legacy
  ratatui/crossterm `ui/` stack.

## Unreleased — buddy v2: account link, hatch celebration, new commands

- **Buddies ride the user account.** First login adopts the guest (device)
  egg into the account (`POST /v1/buddy/adopt`,
  `{"user_id":"u-…","from":"web-…"}`) and pins the companion key, so the same
  egg/buddy follows the login on every device. A *started* account egg
  (`messages > 0`) is never overwritten; an untouched one (created by the
  float loading before the adopt lands) is replaced — later logins no-op
  safely. Store rules covered in `tests/test_buddy_commands.py`.
- **Hatching lives outside the chat.** Chat/AG-UI `/buddy` replies point at
  the float ("Egg incubating — see Buddy float …"), the float owns the egg
  art + progress bar, and the first hatch fires a one-time full-screen
  celebration overlay (crack → confetti → reveal; click/Enter/Esc to
  dismiss; remembered per user as `sentinel_buddy_hatch_seen:<id>`).
- **New commands** (also `--buddy …` and bare `buddy …` spellings):
  `/buddy mimic [text]` (joker nakal of the last message or given text),
  `/buddy joker [text]`, `/buddy mood` (emotion sensing on your last
  message — happy/sad/angry/tired/loving/curious/excited + float animation),
  `/buddy stats` (curiosity/calm/loyalty bars), `/buddy who` (user + AERIS-64
  link), `/buddy about` (AERIS-64 × buddy),
  `/buddy dance|wave|happy|sleep|excited`. Command meta now carries
  `buddy_anim` so the float mirrors it; frontend hooks
  `sentinelBuddyAnim/Refresh/Adopt` added.
- **Mood lexicon** extended (`packages/sentinel/buddy/mood.py`): Hinglish +
  English emotion words (`khush/udaas/gussa/thak/tension/pyaar/…`),
  `sense_mood()` powers `mimic`, `mood` and the float's reaction to chat.
- Float polish: AERIS-64 brand header, progress bar, hover lift, branded
  hatch overlay, reduced-motion support. The Bun edge proxies
  `/v1/buddy/adopt`.

## Unreleased — buddy: Pokémon-style egg hatching

- **Egg incubation (backend).** `packages/sentinel/buddy/eggs.py`: small
  stdlib JSON store (`NEUROFIELD_BUDDY_DB`, default `data/buddy/eggs.json` —
  gitignored). One chat message = one step; passive time adds one step per
  `NEUROFIELD_BUDDY_EGG_PERIOD` seconds (default 300). The requirement is
  deterministic per user (6–10 steps) and "hatched" is derived — the
  companion itself stays `hatch_buddy()`-deterministic, so existing devices
  keep their buddy; the egg only delays the reveal. ASCII wobble + crack
  frames included.
- **API.** `POST /v1/buddy/step` (rate-limited) returns the egg snapshot and
  flips to `hatched: true` + the revealed buddy at the goal. `GET/POST
  /v1/buddy` now carry an `egg` block; `/buddy` (and `/buddy egg`) answers
  with incubation status while unhatched. Chat + AG-UI buddy commands now key
  on the caller's `user_id` (chat body / forwardedProps), falling back to API
  key then session id — so the float and `/buddy` agree. Tests:
  `tests/test_buddy_egg.py` (11).
- **Frontend.** `lib/buddy-float.ts` renders the egg (meta
  `Egg · 3/8 steps · next ~4m`, wobble accelerates near hatching), auto-opens
  the float on the first visit (respects an explicit `/buddy off`), and plays
  the crack → reveal animation. `App.send` fires one
  `window.sentinelBuddyStep()` per real user message and chat calls carry
  `user_id` (`stream.ts`); signed-in accounts incubate `u-<account id>` so
  the egg follows the login, guests keep the per-browser device id. The Bun
  edge proxies `/v1/buddy/step`. Tests: `buddy_egg_test.mjs` (12) + live
  `buddy_egg_live.mjs` (16).

## Unreleased — accounts: login / signup for the web client

- **Backend accounts.** `packages/sentinel/serving/accounts.py`: stdlib-only
  user store (SQLite, `NEUROFIELD_AUTH_DB`, default `data/auth/accounts.db` —
  gitignored). Passwords are scrypt-hashed with per-user salts; sessions are
  opaque bearer tokens stored only as SHA-256 hashes
  (`NEUROFIELD_AUTH_TTL_DAYS`, default 30). New routes: `GET /v1/auth/status`,
  `POST /v1/auth/signup|login|logout`, `GET /v1/auth/me`.
  `NEUROFIELD_AUTH=0` disables accounts (old behavior). With
  `NEUROFIELD_AUTH_REQUIRED=1` the chat endpoints (`/v1/chat`,
  `/v1/chat/stream`, `/ag-ui`, `/v1/ag-ui`) additionally demand an account
  bearer token (X-API-Key still works for machines). Tests:
  `tests/test_accounts.py` (9).
- **Frontend login/signup gate.** `src/app/components/AuthGate.tsx`:
  full-screen sign up / log in card with a one-click "Continue as guest"
  (hidden when the server sets `required: true`). The app probes
  `/v1/auth/status` on load; if the auth API is missing (older server,
  offline dev, stubbed tests) it falls back to guest mode so chat is never
  blocked. Sessions live in localStorage `sentinel_auth_v1`; stored tokens
  are re-validated via `/v1/auth/me` and dropped on 401.
- **Per-user chats + account chip.** Chats in `sentinel_chats_v1` carry an
  `owner` (`"guest"` or account id) and the sidebar only shows the current
  identity's chats. Topbar account chip (`#btnUser`) opens a menu with
  log in / log out. `Authorization: Bearer` rides every model call and the
  Bun edge forwards it (`bearerOf` in `src/server.ts`).

## Unreleased — web client: streaming, AG-UI, history, regenerate/edit

- **AG-UI protocol endpoint.** `packages/sentinel/serving/server.py`: `/ag-ui`
  (alias `/v1/ag-ui`) accepts an AG-UI `RunAgentInput` and streams spec-1.0
  events (`RUN_STARTED` → `TEXT_MESSAGE_START/CONTENT/END` → `RUN_FINISHED`,
  `RUN_ERROR` on failure) with the same control gates as `/v1/chat` (buddy →
  Bhutu → tool-first/SI_v1 → Megha gate → model). `TEXT_MESSAGE_END.metadata`
  carries the `sentinel` block (session_id · model · latency_ms · meta_rag).
  Depends on `ag-ui-protocol` (added to the `serve` extra); the endpoint
  returns 501 when the package is absent. Tests: `tests/test_ag_ui.py`.
- **Streaming SSE hardening (`_TokenPump`).** A client disconnect used to
  strand the global model `_lock`: the abandoned response generator stayed
  suspended inside `_token_stream`'s `with _lock`, and every later request
  blocked forever at zero CPU until a restart (the new UI Stop button made
  this easy to hit). The token generator is now driven by a worker thread
  that always finishes or closes it — cancellation releases the lock at the
  next token, and even a fully abandoned stream drains without wedging the
  server. `_prepare_chat` / `_persist` moved to the threadpool in the two
  streaming endpoints. Tests: `tests/test_stream_pump.py`.
- **Frontend chat client** (`frontend/`):
  - token-by-token streaming with transport fallback
    `/v1/ag-ui` → `/v1/chat/stream` → `/v1/chat` (`src/app/lib/stream.ts`);
  - **Stop** button (aborts stream, keeps partial reply + `stopped` marker);
  - **Regenerate / Edit message** actions (reset server session, replay turn);
  - **History sidebar** (`sentinel_chats_v1` in localStorage; switch, delete,
    clear; mobile overlay via `#btnMenu`);
  - edge: unbuffered SSE pass-through (`proxyStream` in `src/proxy.ts`),
    `/ag-ui` + `/v1/ag-ui` routes, `idleTimeout: 255` on Bun.serve so slow
    token gaps are not reset (default 10s did reset them).
  - Legacy DOM ids/classes are unchanged; the jsdom e2e suite (34 checks)
    still passes.


## Unreleased — Auto RAG (metacognition-controlled)

- **Auto RAG** is the default retrieval path. `src/neurofield/rag/meta_rag.py`:
  - `MetaRAGController` modes: `auto` (default) | `meta` | `always` | `off`
  - `auto_rag(prompt)` one-call API used by chat server + agent loop
  - Live metacognition signals when available; prompt heuristic otherwise
  - Dynamic `top_k`, hysteresis; env: `NEUROFIELD_RAG_MODE`, `NEUROFIELD_META_RAG`,
    `NEUROFIELD_META_RAG_CONF`
- Chat: `use_meta_rag` / `force_rag` on request; `meta_rag` block on response
- Agent: `AgentConfig.use_auto_rag=True` runs retrieval before generate
- Tests: `tests/test_meta_rag.py`

## Unreleased — checkpoint compat + per-user memory: turn-end consolidation

- **Demo checkpoint loads again.** `slow_scale` (new param, init 1.5) was missing from `checkpoints/examples/AERIS_main`, so
  `load_model_from_checkpoint` raised "weights missing from checkpoint" and `test_demo_checkpoint_loads_and_generates` failed on `main`.
  `slow_scale` is now a compat-missing param; a checkpoint without it loads with **1.0** (it was trained with the slow read unscaled),
  a checkpoint that has it keeps its own value.

- After the decode-parity fix `forward()` writes the slow ring only at global segment boundaries (no end-of-call flush), so a chat
  turn shorter than `segment_len` left the user's slow memory empty (measured: 0 slots for a 14-token turn).
  `SENTINEL.consolidate_turn(state)` snapshots the final fast memory into the ring once per turn (paper v3, Section 11);
  the server calls it in `_persist` (skipped for anomaly-flagged turns). Training/decode parity is unaffected (it is not part of `forward`).

## Unreleased — train == generate (decode parity) fix

### Fixed (H1: decode and training were different functions)
- **Error term at decode.** `E = X - shift(X_hat)` shifted inside ONE call, so a 1-token decode step had `shift(X_hat) = 0` and
  `E = X` (the whole embedding) instead of `E_t = X_t - f_p(H_{t-1})`. The previous position's prediction is now recomputed from the
  per-pass `Hin` carried in the state.
- **Fast memory in depth passes >= 1.** Only pass 0 was warm-started from the old `M` (and only when `T == 1`); every other pass
  restarted from zero at each decode step, so memory never accumulated across decode steps for `k_depth > 1`. Every pass now carries
  its own `M_pass[j]`, plus `D_pass[j]` (next write key) and `R_pass[j]` (field input `r_{t-1}`).
- Result: one whole-sequence call == chunked calls == token-by-token decode (max logit diff 0.0 for `k_depth` 1..3; before the fix
  0.50 at `k_depth=1` and 11.03 at `k_depth=3`). Training-form outputs on a fresh sequence are bit-identical to before.
- Slow-ring writes again use one snapshot per `segment_len` tokens (counted across calls) + one at the end of a multi-token call, so
  a training call and decoding write the same slots.
- The audit again carries `router_entropy`, `surprises`, `ks`, `final_mem_norm` (the safety monitor's z-score on router entropy had
  silently lost its input).
- New `tests/test_decode_parity.py` (19 tests, incl. "greedy generation == teacher-forced training forward").

### Known remaining difference (not changed)
- Slow ring READ: a decode step reads snapshots written earlier in the same session; a single training call reads the ring as it
  stood at the start of the call. With the ring on, one-shot vs decode differ (logit diff ~2.7 in a 30-token check). Fix options:
  train with state-carrying windows, or freeze the ring read at call start when serving.

## Unreleased — per-user memory, parallel training, paper v3

### Per-user memory (paper v3 Section 11, ring variant)
- `serving/user_memory.py`: lossless per-chat transcripts + persisted slow ring (safetensors, atomic), per API key / `user_id`.
  Fast memory is never persisted. Chat rehydrates its window from the transcript; re-fed history is replayed with
  `write_slow=False` so the ring is not flooded; ring overflow now **compacts** (count-weighted fold) instead of overwriting
  (`slow_compact`, default on; `filled` holds counts, `filled.sum()` = total writes). Norm cap, audit log, anomaly gating,
  `GET/DELETE /v1/memory`, `GET /v1/memory/transcript`, `remember=false`.
- `SENTINEL.new_state(..., slow=)`, `export_slow_state()`, `forward(write_slow=)`.

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

- Prior SENTINEL / AERIS public tree (audits, 1B config recipe, serving stack)
