"""Custom skills: reusable instruction packs (brand voice, folder layout, workflows).

Skills are YAML/JSON files under the configured skills directory. Connectors
to external apps (Drive, Gmail, GitHub) are registered as named callables by
the serving layer — this module only stores and loads skill definitions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional

from neurofield.ajeebwork.config import load_ajeebwork_config

_CONNECTORS: dict[str, Callable[..., str]] = {}


def _skills_dir() -> Path:
    cfg = load_ajeebwork_config()
    d = Path(cfg.skills_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_skill(name: str, instructions: str, *, metadata: Optional[dict[str, Any]] = None) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)[:64]
    if not safe:
        return "Skills: invalid name."
    path = _skills_dir() / f"{safe}.json"
    payload = {
        "name": name,
        "instructions": instructions,
        "metadata": metadata or {},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return f"Skills saved → {path}"


def load_skill(name: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)[:64]
    path = _skills_dir() / f"{safe}.json"
    if not path.is_file():
        return f"Skills: not found '{name}'."
    data = json.loads(path.read_text(encoding="utf-8"))
    return (
        f"Skill: {data.get('name')}\n"
        f"{data.get('instructions', '')}\n"
        f"metadata={data.get('metadata')}"
    )


def list_skills() -> str:
    paths = sorted(_skills_dir().glob("*.json"))
    if not paths:
        return "Skills: none registered."
    lines = ["Skills:"]
    for p in paths:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            lines.append(f"  - {data.get('name', p.stem)}")
        except Exception:
            lines.append(f"  - {p.stem} (unreadable)")
    return "\n".join(lines)


def register_connector(name: str, fn: Callable[..., str]) -> None:
    _CONNECTORS[name] = fn


def call_connector(name: str, **kwargs: Any) -> str:
    fn = _CONNECTORS.get(name)
    if fn is None:
        known = ", ".join(sorted(_CONNECTORS)) or "(none)"
        return f"Connector '{name}' not registered. Known: {known}"
    try:
        return fn(**kwargs)
    except Exception as e:
        return f"Connector '{name}' error: {type(e).__name__}: {e}"
