"""Shared memory bridge between chat sessions and AjeebWork runs.

When enabled, a session can read/write a small key-value + transcript buffer
that both the chat path and multi-step agent can see. Per-task opt-out via
``memory_enabled=False``.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Optional

from sentinel.ajeebwork.config import load_ajeebwork_config

_lock = threading.Lock()
_STORE: dict[str, dict[str, Any]] = {}


def _session_path(session_id: str) -> Path:
    cfg = load_ajeebwork_config()
    root = Path(cfg.task_store_dir) / "shared_memory"
    root.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id)[:64]
    return root / f"{safe}.json"


def load_session(session_id: str) -> dict[str, Any]:
    with _lock:
        if session_id in _STORE:
            return dict(_STORE[session_id])
    path = _session_path(session_id)
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            with _lock:
                _STORE[session_id] = data
            return dict(data)
        except Exception:
            pass
    return {"session_id": session_id, "kv": {}, "transcript": [], "updated_at": 0.0}


def save_session(session_id: str, data: dict[str, Any]) -> None:
    data = dict(data)
    data["session_id"] = session_id
    data["updated_at"] = time.time()
    with _lock:
        _STORE[session_id] = data
    path = _session_path(session_id)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def memory_get(session_id: str, key: str, *, memory_enabled: bool = True) -> str:
    cfg = load_ajeebwork_config()
    if not cfg.shared_memory_enabled or not memory_enabled:
        return "SharedMemory: disabled for this task."
    data = load_session(session_id)
    val = (data.get("kv") or {}).get(key)
    if val is None:
        return f"SharedMemory: key '{key}' not set."
    return f"SharedMemory {key}={val}"


def memory_set(
    session_id: str,
    key: str,
    value: str,
    *,
    memory_enabled: bool = True,
) -> str:
    cfg = load_ajeebwork_config()
    if not cfg.shared_memory_enabled or not memory_enabled:
        return "SharedMemory: disabled for this task."
    data = load_session(session_id)
    kv = dict(data.get("kv") or {})
    kv[str(key)[:128]] = str(value)[:4000]
    data["kv"] = kv
    save_session(session_id, data)
    return f"SharedMemory set {key}"


def memory_append_transcript(
    session_id: str,
    role: str,
    text: str,
    *,
    memory_enabled: bool = True,
) -> str:
    cfg = load_ajeebwork_config()
    if not cfg.shared_memory_enabled or not memory_enabled:
        return "SharedMemory: disabled for this task."
    data = load_session(session_id)
    tr = list(data.get("transcript") or [])
    tr.append({"role": role, "text": text[:4000], "ts": time.time()})
    data["transcript"] = tr[-200:]  # bound
    save_session(session_id, data)
    return f"SharedMemory transcript +1 (len={len(data['transcript'])})"


def memory_report(session_id: str, *, memory_enabled: bool = True) -> str:
    cfg = load_ajeebwork_config()
    if not cfg.shared_memory_enabled or not memory_enabled:
        return "SharedMemory: disabled."
    data = load_session(session_id)
    kv = data.get("kv") or {}
    tr = data.get("transcript") or []
    lines = [f"SharedMemory session={session_id} keys={len(kv)} turns={len(tr)}"]
    for k, v in list(kv.items())[:20]:
        lines.append(f"  {k}={str(v)[:120]}")
    return "\n".join(lines)
