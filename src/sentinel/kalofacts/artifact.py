"""Core Kalofact model: versioned, editable, dual-view content unit."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

from sentinel.kalofacts.types import ArtifactKind, detect_kind, PREVIEW_KINDS


@dataclass
class ArtifactVersion:
    version: int
    content: str
    created_at: float
    note: str = ""


@dataclass
class Kalofact:
    artifact_id: str
    title: str
    kind: str
    content: str
    session_id: str = "default"
    view_mode: str = "preview"  # preview | code
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    versions: list[dict[str, Any]] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)  # persistent UI/app state
    share_token: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Kalofact":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    def supports_preview(self) -> bool:
        try:
            return ArtifactKind(self.kind) in PREVIEW_KINDS
        except ValueError:
            return False


def create_kalofact(
    title: str,
    content: str,
    *,
    kind: str | None = None,
    session_id: str = "default",
    metadata: Optional[dict[str, Any]] = None,
) -> Kalofact:
    detected = detect_kind(content, kind)
    now = time.time()
    aid = uuid.uuid4().hex[:12]
    art = Kalofact(
        artifact_id=aid,
        title=(title or "Untitled").strip()[:200],
        kind=detected.value,
        content=content or "",
        session_id=session_id,
        view_mode="preview" if detected in PREVIEW_KINDS else "code",
        created_at=now,
        updated_at=now,
        versions=[
            {
                "version": 1,
                "content": content or "",
                "created_at": now,
                "note": "initial",
            }
        ],
        metadata=dict(metadata or {}),
    )
    return art
