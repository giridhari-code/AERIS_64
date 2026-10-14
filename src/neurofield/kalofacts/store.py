"""Persistent Kalofacts library (session + global index)."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Optional

from neurofield.kalofacts.artifact import Kalofact, create_kalofact
from neurofield.kalofacts.config import load_kalofacts_config, store_root

_lock = threading.Lock()
_CACHE: dict[str, Kalofact] = {}


def _path_for(artifact_id: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in artifact_id)[:64]
    return store_root() / f"{safe}.json"


def _index_path() -> Path:
    return store_root() / "index.json"


def _load_index() -> list[dict[str, Any]]:
    p = _index_path()
    if not p.is_file():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_index(entries: list[dict[str, Any]]) -> None:
    _index_path().write_text(json.dumps(entries, indent=2), encoding="utf-8")


def save(art: Kalofact) -> Kalofact:
    cfg = load_kalofacts_config()
    if not cfg.enabled:
        raise RuntimeError("kalofacts disabled")
    if len(art.content.encode("utf-8")) > cfg.max_content_bytes:
        raise ValueError(
            f"content exceeds max_content_bytes ({cfg.max_content_bytes})"
        )
    with _lock:
        path = _path_for(art.artifact_id)
        path.write_text(json.dumps(art.to_dict(), indent=2), encoding="utf-8")
        _CACHE[art.artifact_id] = art
        entries = [e for e in _load_index() if e.get("artifact_id") != art.artifact_id]
        entries.insert(
            0,
            {
                "artifact_id": art.artifact_id,
                "title": art.title,
                "kind": art.kind,
                "session_id": art.session_id,
                "updated_at": art.updated_at,
            },
        )
        _save_index(entries[: 500])
    return art


def get(artifact_id: str) -> Optional[Kalofact]:
    with _lock:
        if artifact_id in _CACHE:
            return _CACHE[artifact_id]
    path = _path_for(artifact_id)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        art = Kalofact.from_dict(data)
        with _lock:
            _CACHE[artifact_id] = art
        return art
    except Exception:
        return None


def list_artifacts(
    session_id: Optional[str] = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    entries = _load_index()
    if session_id:
        entries = [e for e in entries if e.get("session_id") == session_id]
    return entries[:limit]


def delete(artifact_id: str) -> bool:
    with _lock:
        _CACHE.pop(artifact_id, None)
        path = _path_for(artifact_id)
        if path.is_file():
            path.unlink()
        entries = [e for e in _load_index() if e.get("artifact_id") != artifact_id]
        _save_index(entries)
        return True


def create_and_save(
    title: str,
    content: str,
    *,
    kind: str | None = None,
    session_id: str = "default",
    metadata: Optional[dict[str, Any]] = None,
) -> Kalofact:
    cfg = load_kalofacts_config()
    if not cfg.enabled:
        raise RuntimeError("kalofacts disabled")
    existing = list_artifacts(session_id=session_id, limit=10_000)
    if len(existing) >= cfg.max_artifacts_per_session:
        raise RuntimeError(
            f"session artifact limit reached ({cfg.max_artifacts_per_session})"
        )
    art = create_kalofact(
        title, content, kind=kind, session_id=session_id, metadata=metadata
    )
    return save(art)


def should_auto_open(content: str) -> bool:
    cfg = load_kalofacts_config()
    return cfg.enabled and len(content or "") >= cfg.auto_trigger_chars
