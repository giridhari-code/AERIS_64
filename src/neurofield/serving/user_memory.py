"""Per-user long-term memory: lossless chat transcripts + a persisted slow-memory ring.

Division of labour (by design):
  * FAST memory (M, h, dendrite history) belongs to the MODEL's own working state. It is never
    written to disk and starts from zero every turn / session.
  * SLOW memory belongs to the USER. Two parts survive restarts, TTL eviction and new chat windows:
      1. ``chats/<session>.jsonl`` - every turn, verbatim, append-only (never truncated; rotated by
         size into ``<session>.<n>.jsonl``, never deleted except by an explicit delete call);
      2. ``slow.safetensors``      - the slow ring (count-weighted, compacted: old context is
         diluted, not erased) shared by all of the user's chat windows.

Isolation: everything lives under ``<root>/<sha256(api key)[:16]>/<user>-<digest>/``. Opt-in via
``NEUROFIELD_MEMORY_DIR`` and refused without API keys (server side), because transcripts are
personal data. Single-process safe (one lock); use one worker or a shared lock for multi-worker.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Optional

MAX_TURN_CHARS = 20_000


def _slug(value: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in value)[:40] or "default"
    return f"{safe}-{hashlib.sha256(value.encode('utf-8', errors='replace')).hexdigest()[:10]}"


class UserMemoryStore:
    def __init__(self, root: str | os.PathLike[str], max_chat_mb: float = 10.0) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_chat_bytes = int(max(0.01, max_chat_mb) * 1024 * 1024)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ paths
    def _user_dir(self, owner: Optional[str], user_id: str) -> Path:
        o = hashlib.sha256((owner or "").encode()).hexdigest()[:16]
        return self.root / o / _slug(user_id)

    def _chat_files(self, d: Path, session_id: str) -> list[Path]:
        """Oldest -> newest: <s>.1.jsonl, <s>.2.jsonl, ..., <s>.jsonl"""
        base = _slug(session_id)
        rotated = sorted(
            (p for p in (d / "chats").glob(f"{base}.*.jsonl")),
            key=lambda p: int(p.name.split(".")[-2]),
        )
        cur = d / "chats" / f"{base}.jsonl"
        return rotated + ([cur] if cur.exists() else [])

    # ------------------------------------------------------------------ transcripts
    def append_turn(self, owner: Optional[str], user_id: str, session_id: str, role: str, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        rec = json.dumps({"ts": round(time.time(), 3), "role": role, "text": text[:MAX_TURN_CHARS]}, ensure_ascii=False)
        with self._lock:
            d = self._user_dir(owner, user_id)
            (d / "chats").mkdir(parents=True, exist_ok=True, mode=0o700)
            cur = d / "chats" / f"{_slug(session_id)}.jsonl"
            if cur.exists() and cur.stat().st_size + len(rec) > self.max_chat_bytes:
                n = len(self._chat_files(d, session_id))  # rotate, never delete
                cur.rename(d / "chats" / f"{_slug(session_id)}.{n}.jsonl")
            with open(cur, "a", encoding="utf-8") as f:
                f.write(rec + "\n")
                f.flush()
                os.fsync(f.fileno())

    def _read_all(self, d: Path, session_id: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for p in self._chat_files(d, session_id):
            with open(p, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue  # a torn last line after a crash must not hide the rest
        return out

    def transcript(self, owner: Optional[str], user_id: str, session_id: str, limit: int = 200) -> list[dict[str, Any]]:
        with self._lock:
            turns = self._read_all(self._user_dir(owner, user_id), session_id)
        return turns[-max(1, int(limit)) :]

    def recent_turns(self, owner: Optional[str], user_id: str, session_id: str, n: int) -> list[tuple[str, str]]:
        return [(t["role"], t["text"]) for t in self.transcript(owner, user_id, session_id, limit=n)]

    # ------------------------------------------------------------------ slow ring
    def save_slow(self, owner: Optional[str], user_id: str, slow: dict) -> None:
        from safetensors.torch import save_file

        with self._lock:
            d = self._user_dir(owner, user_id)
            d.mkdir(parents=True, exist_ok=True, mode=0o700)
            tmp = d / "slow.safetensors.tmp"
            save_file({k: v.contiguous() for k, v in slow.items()}, str(tmp))
            os.replace(tmp, d / "slow.safetensors")  # atomic: a crash never leaves a half-written ring

    def load_slow(self, owner: Optional[str], user_id: str) -> Optional[dict]:
        from safetensors.torch import load_file

        with self._lock:
            f = self._user_dir(owner, user_id) / "slow.safetensors"
            if not f.exists():
                return None
            try:
                return load_file(str(f))
            except Exception:  # corrupt / foreign file: start empty rather than fail the chat
                return None

    def append_audit(self, owner: Optional[str], user_id: str, record: dict[str, Any]) -> None:
        """One JSON line per consolidation (paper Section 11: 'with a norm cap and an audit log')."""
        with self._lock:
            d = self._user_dir(owner, user_id)
            d.mkdir(parents=True, exist_ok=True, mode=0o700)
            with open(d / "audit.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"ts": round(time.time(), 3), **record}) + "\n")

    # ------------------------------------------------------------------ management
    def info(self, owner: Optional[str], user_id: str) -> dict[str, Any]:
        d = self._user_dir(owner, user_id)
        with self._lock:
            chats = list((d / "chats").glob("*.jsonl")) if (d / "chats").exists() else []
            slow = d / "slow.safetensors"
            return {
                "user_id": user_id,
                "chat_files": len(chats),
                "chat_bytes": sum(p.stat().st_size for p in chats),
                "slow_memory_saved": slow.exists(),
                "slow_memory_bytes": slow.stat().st_size if slow.exists() else 0,
            }

    def delete_user(self, owner: Optional[str], user_id: str) -> bool:
        with self._lock:
            d = self._user_dir(owner, user_id)
            existed = d.exists()
            shutil.rmtree(d, ignore_errors=True)
            return existed


def cap_slow_norm(slow: dict, max_norm: float) -> tuple[dict, int, float]:
    """Scale down every slot whose Frobenius norm exceeds ``max_norm`` (0 = no cap).

    Returns (slow, number_of_slots_capped, largest_norm_before_capping).
    """
    slots = slow["slow_slots"]  # (S, d_k, d_v)
    norms = slots.flatten(1).norm(dim=1)
    biggest = float(norms.max().item()) if norms.numel() else 0.0
    if max_norm <= 0:
        return slow, 0, biggest
    scale = (max_norm / norms.clamp(min=1e-8)).clamp(max=1.0)
    capped = int((scale < 1.0).sum().item())
    if capped:
        slow = dict(slow, slow_slots=slots * scale.view(-1, 1, 1))
    return slow, capped, biggest
