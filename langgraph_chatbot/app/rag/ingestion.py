"""
app/rag/ingestion.py
─────────────────────
Handles document loading, chunking, and storing in ChromaDB.
Supports PDF and plain text files.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.rag.embedder import get_embedder

# ── Config ────────────────────────────────────────────────────────────────────
VECTORSTORE_DIR = str(
    Path(__file__).resolve().parents[2] / "data" / "vectorstore")
COLLECTION_NAME = "rag_documents"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def _get_vectorstore() -> Chroma:
    """Return a persistent ChromaDB vectorstore."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embedder(),
        persist_directory=VECTORSTORE_DIR,
    )


def ingest_documents(file_paths: List[str]) -> int:
    """
    Load, chunk, embed and store documents.
    Returns the number of chunks stored.
    """
    all_docs = []

    for file_path in file_paths:
        path = Path(file_path)
        try:
            if path.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(path))
            else:
                loader = TextLoader(str(path), encoding="utf-8")

            docs = loader.load()

            # Tag each chunk with its source filename
            for doc in docs:
                doc.metadata["source"] = path.name

            all_docs.extend(docs)
        except Exception as e:
            print(f"[RAG] Failed to load {path.name}: {e}")

    if not all_docs:
        return 0

    # Chunk documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(all_docs)

    # Store in ChromaDB
    vectorstore = _get_vectorstore()
    vectorstore.add_documents(chunks)

    print(f"[RAG] Stored {len(chunks)} chunks from {len(file_paths)} file(s)")
    return len(chunks)


def list_ingested_files() -> List[str]:
    """Return unique source filenames currently in the vectorstore."""
    try:
        vectorstore = _get_vectorstore()
        results = vectorstore.get()
        sources = set()
        for meta in results.get("metadatas", []):
            if meta and "source" in meta:
                sources.add(meta["source"])
        return sorted(sources)
    except Exception:
        return []


def clear_vectorstore() -> None:
    """Delete all documents from the vectorstore."""
    try:
        vectorstore = _get_vectorstore()
        vectorstore.delete_collection()
        print("[RAG] Vectorstore cleared")
    except Exception as e:
        print(f"[RAG] Failed to clear vectorstore: {e}")
