# Patch notes — security + train/serve consistency (2026-10)

## Fixed (with tests in `tests/test_sandbox_security.py`, `tests/test_server.py`, `tests/test_slow_context_ring.py`)
| Problem | Fix |
|---|---|
| allow-list only checked first word; `echo hi; <anything>` ran | no shell, argv parsed with shlex |
| `bash`/`sh` allowed | removed |
| no NPROC/FSIZE/NOFILE limits | added (NPROC not enforced for uid 0) |
| "no network" was only proxy env vars | network namespace via `unshare`, `require_isolation` to fail closed |
| `write_file` prefix check escape | `Path.is_relative_to` |
| session id truncated to 64 chars => shared dir | digest of full id in dir name |
| sandbox dirs never deleted | removed on evict/drop |
| tools open when no API key | 503 unless `NEUROFIELD_TOOLS_ALLOW_OPEN=1` |
| any key could read/reset any sandbox | sandbox id = sha256(key, session) |
| `/v1/tools/shell|python|reset` returned 422 for everyone (local request models) | models moved to module level |
| tools not rate-limited | `_limit()` applied |
| default tool session id `default` rejected | `default-session` |
| slow-ring written on every decode token | carried `seg_pos` |
| batch-mean slot weighting (B=1 != B>1) | per-sequence relative surprise |
| `convert_pt_to_safetensors` unusable | restricted `torch.load` |
| `ast.Num` (removed in py3.14), place regex, license metadata, versions, Makefile paths | fixed |

## NOT changed (needs design/experiments, not a patch)
- Slot "key" = row-norm of M (non-negative) vs signed query: verify retrieval with an ablation before touching.
- `/v1/chat` resets recurrent state each turn (memory unused in chat).
- Global model lock is held while streaming (slow client blocks others).
- Per-token Python loop (no parallel scan) — limits scale.
- Sandbox is defence in depth only (`python3` is allowed): run in a container without secrets/network.
