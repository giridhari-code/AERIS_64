"""RAG tool: retrieve + optional ingest for the agent / orchestrator."""

from __future__ import annotations

from typing import Optional

from sentinel.rag.pipeline import get_default_rag


def rag_retrieve(query: str, top_k: int = 4) -> str:
    rag = get_default_rag()
    hits = rag.retrieve(query, top_k=top_k)
    if not hits:
        return "RAG: no chunks in knowledge base (add files under data/knowledge/)."
    lines = [f"RAG retrieved {len(hits)} chunk(s) (store size={len(rag.store)}):"]
    for i, h in enumerate(hits, 1):
        lines.append(f"  [{i}] score={h['score']} src={h.get('source')}: {h['text'][:300]}")
    return "\n".join(lines)


def rag_ingest_text(text: str, source: str = "api") -> str:
    rag = get_default_rag()
    n = rag.ingest_text(text, source=source)
    return f"RAG: ingested {n} new chunk(s); total={len(rag.store)}"


def rag_augment(prompt: str, top_k: int = 4) -> tuple[str, str]:
    """Returns (augmented_prompt, note_for_logs)."""
    rag = get_default_rag()
    aug, hits = rag.augment(prompt, top_k=top_k)
    note = f"RAG augmented with {len(hits)} chunk(s)" if hits else "RAG: no hits"
    return aug, note
