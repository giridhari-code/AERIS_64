"""GK Cache — General Knowledge cache for SENTINEL.

Session + process level cache of retrieved / injected knowledge so inference
does not re-hit RAG / external stores on every token or every turn.

Design (recurrent-native, not Transformer KV):
  • Fixed capacity (default 128 entries) — bounded memory
  • Hash embedding + optional torch vectors for similarity
  • Hit → skip expensive retrieve; soft-inject into J-space / prompt context
  • TTL + LRU eviction
  • Self-heal: drop NaN / empty entries
  • Works beside InferenceCache (session state) and Fast/Slow memory

Integration
-----------
Populate via ``put`` / ``put_many`` from RAG or curated knowledge.
Query via ``lookup`` / ``format_context`` during serving.
``broadcast_vector`` yields a fixed-size bias for field/router injection.
Capacity, embedding dim, and min_score are constructor arguments (wire from ModelConfig).
"""

from __future__ import annotations

import hashlib
import math
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from torch import Tensor


def _tokens(text: str) -> list[str]:
    import re
    return [t.lower() for t in re.findall(r"[a-zA-Z0-9_']+|[^\s]", text or "") if t.strip()]


def hash_embed(text: str, dim: int = 256) -> list[float]:
    """Feature hashing unigram+bigram → unit vector (numpy-free)."""
    vec = [0.0] * dim
    toks = _tokens(text)
    feats = list(toks) + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
    for f in feats:
        h = int(hashlib.md5(f.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if (h >> 8) & 1 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def _cos(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    return sum(a[i] * b[i] for i in range(n))


@dataclass
class GKEntry:
    """One knowledge unit in the cache."""

    key: str
    text: str
    source: str = ""
    score: float = 1.0
    vector: list[float] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)
    hits: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def touch(self) -> None:
        self.last_access = time.time()
        self.hits += 1


class GKCache:
    """Fixed-capacity General Knowledge cache for SENTINEL inference."""

    def __init__(
        self,
        capacity: int = 128,
        dim: int = 256,
        ttl_sec: float = 0.0,
        min_score: float = 0.25,
    ):
        self.capacity = max(8, int(capacity))
        self.dim = int(dim)
        self.ttl_sec = float(ttl_sec)  # 0 = no expiry
        self.min_score = float(min_score)
        self._store: OrderedDict[str, GKEntry] = OrderedDict()
        self.stats = {"hits": 0, "misses": 0, "puts": 0, "evictions": 0}

    def __len__(self) -> int:
        return len(self._store)

    def clear(self) -> None:
        self._store.clear()

    def _make_key(self, text: str, source: str = "") -> str:
        raw = f"{source}::{text.strip()[:512]}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]

    def _expire(self) -> None:
        if self.ttl_sec <= 0:
            return
        now = time.time()
        dead = [k for k, e in self._store.items() if now - e.created_at > self.ttl_sec]
        for k in dead:
            self._store.pop(k, None)
            self.stats["evictions"] += 1

    def _evict_lru(self) -> None:
        while len(self._store) >= self.capacity:
            self._store.popitem(last=False)
            self.stats["evictions"] += 1

    def put(
        self,
        text: str,
        *,
        source: str = "",
        score: float = 1.0,
        key: Optional[str] = None,
        meta: Optional[dict] = None,
    ) -> str:
        """Insert or refresh one knowledge entry. Returns cache key."""
        text = (text or "").strip()
        if not text:
            return ""
        self._expire()
        k = key or self._make_key(text, source)
        vec = hash_embed(text, self.dim)
        if k in self._store:
            e = self._store.pop(k)
            e.text = text
            e.source = source or e.source
            e.score = max(e.score, float(score))
            e.vector = vec
            e.touch()
            if meta:
                e.meta.update(meta)
            self._store[k] = e
        else:
            self._evict_lru()
            self._store[k] = GKEntry(
                key=k,
                text=text,
                source=source,
                score=float(score),
                vector=vec,
                meta=dict(meta or {}),
            )
        self.stats["puts"] += 1
        return k

    def put_many(self, chunks: list[dict], *, source: str = "rag") -> int:
        n = 0
        for c in chunks:
            text = str(c.get("text") or c.get("content") or "").strip()
            if not text:
                continue
            self.put(
                text,
                source=str(c.get("source") or source),
                score=float(c.get("score") or 1.0),
                key=str(c["id"]) if c.get("id") else None,
                meta={k: v for k, v in c.items() if k not in ("text", "content", "score", "source", "id")},
            )
            n += 1
        return n

    def lookup(
        self,
        query: str,
        *,
        top_k: int = 4,
        min_score: Optional[float] = None,
    ) -> list[GKEntry]:
        """Similarity search over cached knowledge."""
        self._expire()
        if not self._store or not (query or "").strip():
            self.stats["misses"] += 1
            return []
        q = hash_embed(query, self.dim)
        thr = self.min_score if min_score is None else float(min_score)
        scored: list[tuple[float, GKEntry]] = []
        for e in self._store.values():
            if not e.vector:
                continue
            s = _cos(q, e.vector) * (0.7 + 0.3 * min(1.0, e.score))
            if s >= thr:
                scored.append((s, e))
        scored.sort(key=lambda x: -x[0])
        out = []
        for s, e in scored[: max(1, top_k)]:
            e.touch()
            # move to end = most recently used
            self._store.move_to_end(e.key)
            out.append(e)
        if out:
            self.stats["hits"] += 1
        else:
            self.stats["misses"] += 1
        return out

    def format_context(self, query: str, top_k: int = 3, max_chars: int = 1500) -> str:
        """Text block suitable for prompt / system injection."""
        hits = self.lookup(query, top_k=top_k)
        if not hits:
            return ""
        parts = []
        total = 0
        for e in hits:
            line = f"- ({e.source or 'gk'}) {e.text.strip()}"
            if total + len(line) > max_chars:
                break
            parts.append(line)
            total += len(line)
        return "\n".join(parts)

    def broadcast_vector(self, d_model: int, query: Optional[str] = None) -> Tensor:
        """Mean of top hit embeddings, projected/padded to ``d_model`` for field bias."""
        if query:
            hits = self.lookup(query, top_k=4)
        else:
            # most recent entries
            hits = list(self._store.values())[-4:]
        if not hits:
            return torch.zeros(d_model)
        acc = [0.0] * self.dim
        for e in hits:
            for i, v in enumerate(e.vector):
                acc[i] += v
        n = float(len(hits))
        acc = [x / n for x in acc]
        t = torch.tensor(acc, dtype=torch.float32)
        if t.numel() >= d_model:
            return t[:d_model]
        out = torch.zeros(d_model)
        out[: t.numel()] = t
        return out

    def heal(self) -> int:
        """Drop empty / corrupt entries. Returns number removed."""
        dead = [k for k, e in self._store.items() if not (e.text or "").strip()]
        for k in dead:
            self._store.pop(k, None)
        self.stats["evictions"] += len(dead)
        return len(dead)

    def summary(self) -> dict[str, float]:
        return {
            "size": float(len(self._store)),
            "capacity": float(self.capacity),
            "hits": float(self.stats["hits"]),
            "misses": float(self.stats["misses"]),
            "puts": float(self.stats["puts"]),
            "evictions": float(self.stats["evictions"]),
            "hit_rate": float(self.stats["hits"] / max(1, self.stats["hits"] + self.stats["misses"])),
        }


# Process-level singleton (optional shared GK across sessions)
_GLOBAL_GK: Optional[GKCache] = None


def get_gk_cache(capacity: int = 128, dim: int = 256) -> GKCache:
    global _GLOBAL_GK
    if _GLOBAL_GK is None:
        _GLOBAL_GK = GKCache(capacity=capacity, dim=dim)
    return _GLOBAL_GK
