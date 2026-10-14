"""Split domain documents into overlapping chunks."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator


def chunk_text(
    text: str,
    *,
    chunk_size: int = 400,
    overlap: int = 80,
    source: str = "",
) -> list[dict]:
    """Split ``text`` into overlapping character windows."""
    text = (text or "").strip()
    if not text:
        return []
    chunks: list[dict] = []
    n = len(text)
    start = 0
    idx = 0
    step = max(1, chunk_size - overlap)
    while start < n:
        end = min(n, start + chunk_size)
        piece = text[start:end].strip()
        if piece:
            chunks.append(
                {
                    "id": f"{source or 'doc'}:{idx}",
                    "text": piece,
                    "source": source,
                    "start": start,
                    "end": end,
                }
            )
            idx += 1
        if end >= n:
            break
        start += step
    return chunks


def load_and_chunk(path: str | Path, **kwargs) -> list[dict]:
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="replace")
    return chunk_text(text, source=str(p.name), **kwargs)


def iter_knowledge_dir(root: str | Path, **kwargs) -> Iterator[dict]:
    root = Path(root)
    if not root.is_dir():
        return
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in {".txt", ".md", ".jsonl"} and p.is_file():
            for c in load_and_chunk(p, **kwargs):
                yield c
