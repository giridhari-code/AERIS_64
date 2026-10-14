"""Export and share helpers for Kalofacts."""

from __future__ import annotations

import json
import secrets
import time
from pathlib import Path
from typing import Optional

from neurofield.kalofacts.config import load_kalofacts_config, store_root
from neurofield.kalofacts.store import get, save


def export_source(artifact_id: str, dest_path: Optional[str] = None) -> str:
    art = get(artifact_id)
    if not art:
        return f"Kalofacts: not found {artifact_id}"
    ext = {
        "markdown": ".md",
        "html": ".html",
        "python": ".py",
        "javascript": ".js",
        "react": ".jsx",
        "sql": ".sql",
        "bash": ".sh",
        "mermaid": ".mmd",
        "svg": ".svg",
        "json": ".json",
    }.get(art.kind, ".txt")
    if dest_path:
        path = Path(dest_path)
    else:
        path = store_root() / "exports" / f"{art.artifact_id}{ext}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(art.content, encoding="utf-8")
    return f"Kalofacts exported → {path}"


def create_share_token(artifact_id: str) -> str:
    cfg = load_kalofacts_config()
    if not cfg.allow_public_share:
        return (
            "Kalofacts: public share disabled. "
            "Set NEUROFIELD_KALOFACTS_PUBLIC_SHARE=1 to enable."
        )
    art = get(artifact_id)
    if not art:
        return f"Kalofacts: not found {artifact_id}"
    token = secrets.token_urlsafe(16)
    art.share_token = token
    art.updated_at = time.time()
    save(art)
    # Local share record (serving layer can map token → public URL)
    share_dir = store_root() / "shares"
    share_dir.mkdir(parents=True, exist_ok=True)
    (share_dir / f"{token}.json").write_text(
        json.dumps(
            {
                "artifact_id": artifact_id,
                "title": art.title,
                "kind": art.kind,
                "created_at": time.time(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return f"Kalofacts share_token={token} artifact_id={artifact_id}"


def resolve_share(token: str) -> str:
    path = store_root() / "shares" / f"{token}.json"
    if not path.is_file():
        return "Kalofacts: invalid or expired share token."
    meta = json.loads(path.read_text(encoding="utf-8"))
    art = get(str(meta.get("artifact_id") or ""))
    if not art or art.share_token != token:
        return "Kalofacts: share not active."
    return (
        f"Kalofacts shared title={art.title} kind={art.kind}\n"
        f"---\n{art.content[:8000]}"
    )
