"""0z session + audit trail (company standard)."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ZeroZTurn:
    role: str
    content: str
    ts: float = field(default_factory=time.time)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ZeroZSession:
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: float = field(default_factory=time.time)
    turns: list[ZeroZTurn] = field(default_factory=list)
    audit: list[dict[str, Any]] = field(default_factory=list)

    def add(self, role: str, content: str, **meta: Any) -> None:
        self.turns.append(ZeroZTurn(role=role, content=content, meta=meta))

    def log(self, event: str, **payload: Any) -> None:
        self.audit.append({"event": event, "ts": time.time(), **payload})

    def transcript(self, max_turns: int = 20) -> str:
        parts = []
        for turn in self.turns[-max_turns:]:
            parts.append(f"{turn.role.upper()}: {turn.content}")
        return "\n".join(parts)

    def save_audit(self, directory: str | Path) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"zeroz_{self.session_id}.jsonl"
        with path.open("a", encoding="utf-8") as f:
            for row in self.audit:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            self.audit.clear()
        return path

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "turns": [asdict(t) for t in self.turns],
        }
