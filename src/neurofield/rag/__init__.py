"""Lightweight RAG pipeline for AERIS_64.

Flow (matches the RAG use-case diagram):
  knowledge base → split/encode → vector store
  user prompt → retrieve similar chunks → rank → augment prompt → LLM

Metacognitive control (Adaptive RAG):
  Metacognition signals (confidence / arousal / conflict) decide *whether*
  and *how much* to retrieve — see ``meta_rag.MetaRAGController``.

No external vector DB required for the default path (in-memory + optional JSON persist).
"""

from neurofield.rag.pipeline import RAGPipeline, get_default_rag
from neurofield.rag.meta_rag import (
    MetaRAGController,
    MetaRAGDecision,
    get_meta_rag,
    estimate_prompt_confidence,
    auto_rag,
)

__all__ = [
    "RAGPipeline",
    "get_default_rag",
    "MetaRAGController",
    "MetaRAGDecision",
    "get_meta_rag",
    "estimate_prompt_confidence",
    "auto_rag",
]
