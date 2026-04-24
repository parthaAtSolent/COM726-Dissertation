"""
app/rag/retriever.py
─────────────────────
Retrieves top-K relevant chunks from ChromaDB for a given query.
"""

from __future__ import annotations

# instead of from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma

from app.rag.ingestion import VECTORSTORE_DIR, COLLECTION_NAME
from app.rag.embedder import get_embedder

TOP_K = 4


def retrieve_context(query: str) -> str:
    """
    Run similarity search against the vectorstore.

    Returns a formatted string of relevant chunks ready to inject into the
    LLM prompt, or an empty string if the vectorstore is empty.

    Raises
    ------
    RuntimeError
        If the vectorstore exists but the similarity search fails (e.g.
        Ollama is not running). The caller in graph.py should surface this
        to the user rather than silently dropping context.
    """
    print(f"[RAG] Retrieving context for query: '{query[:100]}...'")
    print(f"[RAG] vectorstore path: {VECTORSTORE_DIR}")

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embedder(),
        persist_directory=VECTORSTORE_DIR,
    )

    # Check collection is populated — using public API (chromadb 1.x compatible)
    try:
        count = len(vectorstore.get(include=[])["ids"])
        print(f"[RAG] vectorstore has {count} chunks")
    except Exception as e:
        print(f"[RAG] Error checking count: {e}")
        raise RuntimeError(f"Could not read vectorstore: {e}") from e

    if count == 0:
        print("[RAG] vectorstore is empty — skipping retrieval")
        return ""

    print(f"[RAG] Searching {count} chunks for relevant content...")

    # Embed query and search — raises if Ollama is down
    try:
        docs = vectorstore.similarity_search(query, k=TOP_K)
        print(f"[RAG] Similarity search returned {len(docs)} documents")
    except Exception as e:
        print(f"[RAG] Similarity search failed: {e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(
            f"Similarity search failed — is Ollama running with 'nomic-embed-text'?\n{e}"
        ) from e

    if not docs:
        print("[RAG] No relevant chunks found")
        return ""

    print(f"[RAG] Retrieved {len(docs)} chunk(s)")

    # Print scores if available (for debugging)
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        content_preview = doc.page_content[:100]
        print(
            f"[RAG] Chunk {i+1}: source={source}, preview={content_preview}...")

    chunks = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        chunks.append(f"[Source: {source}]\n{doc.page_content}")

    return "\n\n---\n\n".join(chunks)


def get_vectorstore_count() -> int:
    """Return total number of chunks currently stored."""
    try:
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=get_embedder(),
            persist_directory=VECTORSTORE_DIR,
        )
        count = len(vectorstore.get(include=[])["ids"])
        print(f"[RAG] vectorstore count: {count}")
        return count
    except Exception as e:
        print(f"[RAG] get_vectorstore_count failed: {e}")
        return 0
