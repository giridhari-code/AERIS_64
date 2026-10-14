"""In-memory vector store with hashed n-gram embeddings (numpy only)."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Optional

_TOKEN = re.compile(r"[a-zA-Z0-9_']+|[^\s]", re.UNICODE)


def _tokens(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN.findall(text or "") if t.strip()]


def _hash_embed(text: str, dim: int = 256) -> list[float]:
    """Feature hashing over unigrams + bigrams → unit vector."""
    vec = [0.0] * dim
    toks = _tokens(text)
    feats = list(toks)
    feats += [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
    for f in feats:
        h = int(hashlib.md5(f.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if (h >> 8) & 1 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def _cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class VectorStore:
    """Simple continuous-updatable vector DB (in-process)."""

    def __init__(self, dim: int = 256):
        self.dim = dim
        self.ids: list[str] = []
        self.texts: list[str] = []
        self.sources: list[str] = []
        self.vectors: list[list[float]] = []

    def __len__(self) -> int:
        return len(self.ids)

    def add(self, chunks: list[dict]) -> int:
        n = 0
        existing = set(self.ids)
        for c in chunks:
            cid = str(c.get("id") or f"auto:{len(self.ids)}")
            if cid in existing:
                continue
            text = str(c.get("text") or "").strip()
            if not text:
                continue
            self.ids.append(cid)
            self.texts.append(text)
            self.sources.append(str(c.get("source") or ""))
            self.vectors.append(_hash_embed(text, self.dim))
            existing.add(cid)
            n += 1
        return n

    def search(self, query: str, top_k: int = 4) -> list[dict]:
        if not self.vectors:
            return []
        q = _hash_embed(query, self.dim)
        scored = [(_cos(q, v), i) for i, v in enumerate(self.vectors)]
        scored.sort(key=lambda x: -x[0])
        out = []
        for score, i in scored[: max(1, top_k)]:
            out.append(
                {
                    "id": self.ids[i],
                    "text": self.texts[i],
                    "source": self.sources[i],
                    "score": round(float(score), 4),
                }
            )
        return out

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "dim": self.dim,
            "items": [
                {"id": i, "text": t, "source": s, "vector": v}
                for i, t, s, v in zip(self.ids, self.texts, self.sources, self.vectors)
            ],
        }
        path.write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "VectorStore":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        store = cls(dim=int(data.get("dim") or 256))
        for it in data.get("items") or []:
            store.ids.append(it["id"])
            store.texts.append(it["text"])
            store.sources.append(it.get("source") or "")
            store.vectors.append(it["vector"])
        return store
