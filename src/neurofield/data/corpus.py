"""Multi-format corpus loader → unified plain text for NeuroField training.

Supported inputs:
  .txt   — raw lines (User:/Assistant: or plain prose). Lines starting with # skipped.
  .jsonl — one JSON object per line
  .json  — list of objects, or {"data": [...]} / {"messages": [...]}

Recognised object fields (any common SFT schema):
  messages: [{role, content}, ...]     → User:/Assistant: turns
  user + assistant | prompt + response | instruction + output | question + answer
  text | content | body                → plain paragraph

No demo hardcoding — only schema normalization.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

_USER_KEYS = ("user", "prompt", "instruction", "question", "input", "query", "human")
_ASSIST_KEYS = ("assistant", "response", "output", "answer", "completion", "target", "gpt")
_TEXT_KEYS = ("text", "content", "body", "document", "raw")


def _pick(obj: dict[str, Any], keys: Iterable[str]) -> str | None:
    for k in keys:
        if k in obj and obj[k] is not None:
            v = obj[k]
            if isinstance(v, str) and v.strip():
                return v.strip()
            if isinstance(v, (list, dict)):
                return json.dumps(v, ensure_ascii=False)
    return None


def _messages_to_text(messages: list) -> str:
    parts: list[str] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or m.get("from") or m.get("speaker") or "").lower()
        content = m.get("content") or m.get("value") or m.get("text") or ""
        if isinstance(content, list):
            # multimodal-style: take text parts only
            content = " ".join(
                str(p.get("text", p)) if isinstance(p, dict) else str(p) for p in content
            )
        content = str(content).strip()
        if not content:
            continue
        if role in ("user", "human", "prompt", "instruction"):
            parts.append(f"User: {content}")
        elif role in ("assistant", "gpt", "bot", "model", "response"):
            parts.append(f"Assistant: {content}")
        elif role in ("system",):
            parts.append(f"# system: {content}")
        else:
            parts.append(content)
    return "\n".join(parts)


def record_to_text(obj: Any) -> str:
    """Normalize one JSON record to training text (may be multi-line)."""
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj.strip()
    if isinstance(obj, list):
        # list of messages or list of strings
        if obj and isinstance(obj[0], dict) and (
            "role" in obj[0] or "content" in obj[0]
        ):
            return _messages_to_text(obj)
        return "\n".join(record_to_text(x) for x in obj if record_to_text(x))
    if not isinstance(obj, dict):
        return str(obj).strip()

    if "messages" in obj and isinstance(obj["messages"], list):
        return _messages_to_text(obj["messages"])

    user = _pick(obj, _USER_KEYS)
    assistant = _pick(obj, _ASSIST_KEYS)
    if user and assistant:
        return f"User: {user}\nAssistant: {assistant}"
    if user and not assistant:
        return f"User: {user}"
    if assistant and not user:
        return f"Assistant: {assistant}"

    text = _pick(obj, _TEXT_KEYS)
    if text:
        return text
    return ""


def load_txt(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    lines = [ln for ln in raw.splitlines() if not ln.strip().startswith("#")]
    return "\n".join(lines).strip()


def load_jsonl(path: Path) -> str:
    blocks: list[str] = []
    with path.open(encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("%s:%d skip invalid JSON: %s", path.name, i, e)
                continue
            t = record_to_text(obj)
            if t:
                blocks.append(t)
    return "\n\n".join(blocks)


def load_json(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    if isinstance(data, dict):
        for key in ("data", "examples", "items", "conversations", "messages"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
        else:
            # single record
            t = record_to_text(data)
            return t
    if not isinstance(data, list):
        return record_to_text(data)
    blocks = [record_to_text(x) for x in data]
    blocks = [b for b in blocks if b]
    return "\n\n".join(blocks)


def load_corpus_file(path: str | Path) -> str:
    """Load one file of any supported type into plain training text."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    suf = path.suffix.lower()
    if suf == ".txt":
        return load_txt(path)
    if suf == ".jsonl":
        return load_jsonl(path)
    if suf == ".json":
        return load_json(path)
    # unknown: try text, then jsonl
    try:
        return load_txt(path)
    except Exception:
        return load_jsonl(path)


def load_corpus(paths: list[str | Path]) -> tuple[str, list[dict[str, Any]]]:
    """Load many files; return (merged_text, manifest)."""
    parts: list[str] = []
    manifest: list[dict[str, Any]] = []
    for p in paths:
        path = Path(p)
        if not path.is_file():
            logger.warning("skip missing data file: %s", path)
            continue
        text = load_corpus_file(path)
        n = len(text)
        parts.append(text)
        manifest.append({"path": str(path), "bytes": path.stat().st_size, "chars": n, "format": path.suffix.lower() or ".txt"})
        logger.info("loaded %s (%s, %d chars)", path, path.suffix or ".txt", n)
    merged = "\n\n".join(t for t in parts if t).strip() + "\n"
    return merged, manifest
