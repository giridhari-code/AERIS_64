"""RAG pipeline: ingest → retrieve → rank → augment prompt."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from neurofield.rag.chunker import chunk_text, iter_knowledge_dir, load_and_chunk
from neurofield.rag.store import VectorStore

_DEFAULT: Optional["RAGPipeline"] = None


class RAGPipeline:
    def __init__(self, dim: int = 256, chunk_size: int = 400, overlap: int = 80):
        self.store = VectorStore(dim=dim)
        self.chunk_size = chunk_size
        self.overlap = overlap

    def ingest_text(self, text: str, source: str = "inline") -> int:
        chunks = chunk_text(
            text, chunk_size=self.chunk_size, overlap=self.overlap, source=source
        )
        return self.store.add(chunks)

    def ingest_file(self, path: str | Path) -> int:
        chunks = load_and_chunk(
            path, chunk_size=self.chunk_size, overlap=self.overlap
        )
        return self.store.add(chunks)

    def ingest_dir(self, root: str | Path) -> int:
        n = 0
        for c in iter_knowledge_dir(
            root, chunk_size=self.chunk_size, overlap=self.overlap
        ):
            n += self.store.add([c])
        return n

    def retrieve(self, query: str, top_k: int = 4) -> list[dict]:
        return self.store.search(query, top_k=top_k)

    def augment(self, prompt: str, top_k: int = 4) -> tuple[str, list[dict]]:
        """Retrieve + rank chunks, return (augmented_prompt, hits)."""
        hits = self.retrieve(prompt, top_k=top_k)
        if not hits:
            return prompt, []
        block_lines = []
        for i, h in enumerate(hits, 1):
            src = h.get("source") or "kb"
            block_lines.append(f"[{i}] ({src}, score={h['score']}) {h['text']}")
        block = "\n".join(block_lines)
        augmented = (
            "[Retrieved context — use these facts; prefer them over guessing]\n"
            f"{block}\n\n"
            f"User question: {prompt}"
        )
        return augmented, hits

    def stats(self) -> dict:
        return {"chunks": len(self.store), "dim": self.store.dim}


def get_default_rag(reload: bool = False) -> RAGPipeline:
    """Process-wide RAG instance. Auto-loads data/knowledge + sample corpora."""
    global _DEFAULT
    if _DEFAULT is not None and not reload:
        return _DEFAULT

    rag = RAGPipeline()
    # Domain knowledge dir (user can drop .txt / .md here)
    for root in (
        Path("data/knowledge"),
        Path(__file__).resolve().parents[3] / "data" / "knowledge",
    ):
        if root.is_dir():
            rag.ingest_dir(root)
            break

    # Seed with project samples so RAG works out of the box
    samples = Path("data/samples")
    if not samples.is_dir():
        samples = Path(__file__).resolve().parents[3] / "data" / "samples"
    if samples.is_dir():
        for name in ("company_corpus.txt", "india_multilang.txt", "skills_real.txt"):
            p = samples / name
            if p.is_file():
                rag.ingest_file(p)

    # Optional persist path
    persist = os.environ.get("NEUROFIELD_RAG_STORE")
    if persist and Path(persist).is_file():
        try:
            rag.store = VectorStore.load(persist)
        except Exception:
            pass

    _DEFAULT = rag
    return _DEFAULT
