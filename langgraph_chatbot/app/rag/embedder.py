"""
app/rag/embedder.py
────────────────────
Singleton OllamaEmbeddings using nomic-embed-text.
Runs entirely locally — no API key needed.
"""
from __future__ import annotations
from langchain_ollama import OllamaEmbeddings

_embedder: OllamaEmbeddings | None = None


def get_embedder() -> OllamaEmbeddings:
    """Return a shared embedder instance (lazy init)."""
    global _embedder
    if _embedder is None:
        _embedder = OllamaEmbeddings(model="nomic-embed-text")
    return _embedder
