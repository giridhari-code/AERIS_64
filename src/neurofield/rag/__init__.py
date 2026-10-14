"""Lightweight RAG pipeline for AERIS_64.

Flow (matches the RAG use-case diagram):
  knowledge base → split/encode → vector store
  user prompt → retrieve similar chunks → rank → augment prompt → LLM

No external vector DB required for the default path (in-memory + optional JSON persist).
"""

from neurofield.rag.pipeline import RAGPipeline, get_default_rag

__all__ = ["RAGPipeline", "get_default_rag"]
