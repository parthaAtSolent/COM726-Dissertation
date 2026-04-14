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

# instead of from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.rag.embedder import get_embedder

# ── Config ────────────────────────────────────────────────────────────────────
# FIXED: Use the actual path with capital 'V' in vectorstore
vECTORSTORE_DIR = str(
    # Changed from 'vectorstore' to 'vectorstore'
    Path(__file__).resolve().parents[2] / "data" / "vectorstore")
COLLECTION_NAME = "rag_documents"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Debug: Print the vectorstore path
print(f"[RAG] vectorstore directory: {vECTORSTORE_DIR}")
print(f"[RAG] Full path: {Path(vECTORSTORE_DIR).absolute()}")


def _get_vectorstore() -> Chroma:
    """Return a persistent ChromaDB vectorstore."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embedder(),
        persist_directory=vECTORSTORE_DIR,
    )


def ingest_documents(file_paths: List[str]) -> int:
    """
    Load, chunk, embed and store documents.
    Returns the number of chunks stored, or raises on embedding failure.
    """
    print(f"[RAG] Starting ingestion of {len(file_paths)} file(s)")
    print(f"[RAG] Files: {file_paths}")

    all_docs = []

    # ── 1. Load files ─────────────────────────────────────────────────────────
    for file_path in file_paths:
        path = Path(file_path)
        print(f"[RAG] Loading: {path.name} ({path.suffix})")

        try:
            if path.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(path))
                print(f"[RAG] Using PyPDFLoader for {path.name}")
            else:
                loader = TextLoader(str(path), encoding="utf-8")
                print(f"[RAG] Using TextLoader for {path.name}")

            docs = loader.load()
            print(f"[RAG] Loaded {len(docs)} page(s) from {path.name}")

            if not docs:
                print(f"[RAG] Warning: {path.name} produced no content")
                continue

            for doc in docs:
                doc.metadata["source"] = path.name

            all_docs.extend(docs)
            print(f"[RAG] Total docs so far: {len(all_docs)}")

        except Exception as e:
            print(f"[RAG] Failed to load {path.name}: {e}")
            raise RuntimeError(f"Could not read '{path.name}': {e}") from e

    if not all_docs:
        print("[RAG] No content extracted from any file")
        return 0

    # ── 2. Chunk ──────────────────────────────────────────────────────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(all_docs)
    print(f"[RAG] Split into {len(chunks)} chunks")

    # Print sample chunk
    if chunks:
        print(
            f"[RAG] Sample chunk (first 200 chars): {chunks[0].page_content[:200]}...")

    # ── 3. Embed and store ────────────────────────────────────────────────────
    try:
        print(f"[RAG] Getting vectorstore at: {vECTORSTORE_DIR}")
        vectorstore = _get_vectorstore()

        print(f"[RAG] Adding {len(chunks)} documents to vectorstore...")
        vectorstore.add_documents(chunks)
        print(f"[RAG] Successfully added documents")

        # Verify by checking count
        count = vectorstore._collection.count()
        print(f"[RAG] vectorstore now contains {count} total chunks")

    except Exception as e:
        print(f"[RAG] Embedding/storage failed: {e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(
            f"Embedding failed — is Ollama running with 'nomic-embed-text'?\n{e}"
        ) from e

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
    except Exception as e:
        print(f"[RAG] Error listing ingested files: {e}")
        return []


def get_vectorstore_count() -> int:
    """Return total number of chunks currently stored."""
    try:
        count = _get_vectorstore()._collection.count()
        print(f"[RAG] vectorstore count: {count}")
        return count
    except Exception as e:
        print(f"[RAG] Error getting count: {e}")
        return 0


def clear_vectorstore() -> None:
    """Delete all documents from the vectorstore."""
    try:
        vectorstore = _get_vectorstore()
        vectorstore.delete_collection()
        print("[RAG] vectorstore cleared")
    except Exception as e:
        print(f"[RAG] Failed to clear vectorstore: {e}")
