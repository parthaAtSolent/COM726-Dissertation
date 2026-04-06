"""
app/rag/retriever.py
─────────────────────
Retrieves top-K relevant chunks from ChromaDB for a given query.
"""
from __future__ import annotations

from langchain_community.vectorstores import Chroma

from app.rag.ingestion import VECTORSTORE_DIR, COLLECTION_NAME
from app.rag.embedder import get_embedder

TOP_K = 4   # number of chunks to retrieve


def retrieve_context(query: str) -> str:
    """
    Run similarity search against the vectorstore.
    Returns a single string of concatenated relevant chunks,
    ready to be injected into the LLM prompt.
    Returns empty string if vectorstore is empty.
    """
    try:
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=get_embedder(),
            persist_directory=VECTORSTORE_DIR,
        )

        # Check if collection has any documents
        if vectorstore._collection.count() == 0:
            return ""

        docs = vectorstore.similarity_search(query, k=TOP_K)

        if not docs:
            return ""

        # Format chunks with source labels
        chunks = []
        for doc in docs:
            source = doc.metadata.get("source", "unknown")
            chunks.append(f"[Source: {source}]\n{doc.page_content}")

        return "\n\n---\n\n".join(chunks)

    except Exception as e:
        print(f"[RAG] Retrieval error: {e}")
        return ""
