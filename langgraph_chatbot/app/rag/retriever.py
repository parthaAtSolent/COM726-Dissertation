"""
app/rag/retriever.py
─────────────────────
Enhanced retrieval with model-compatible formatting.
"""

from __future__ import annotations

from langchain_chroma import Chroma
from app.rag.ingestion import VECTORSTORE_DIR, COLLECTION_NAME
from app.rag.embedder import get_embedder

TOP_K = 4


def retrieve_context(query: str, model_key: str = None) -> dict:
    """
    Run similarity search and return structured context.

    Returns:
        dict with keys:
        - 'has_context': bool
        - 'raw_chunks': list of dicts with 'content', 'source'
        - 'formatted_text': str ready for prompt
        - 'formatted_markdown': str with XML tags (better for some models)
        - 'count': int number of chunks
    """
    print(f"[RAG] Retrieving context for query: '{query[:100]}...'")
    print(f"[RAG] vectorstore path: {VECTORSTORE_DIR}")

    result = {
        'has_context': False,
        'raw_chunks': [],
        'formatted_text': '',
        'formatted_markdown': '',
        'count': 0
    }

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embedder(),
        persist_directory=VECTORSTORE_DIR,
    )

    # Check if vectorstore has content
    try:
        count = len(vectorstore.get(include=[])["ids"])
        print(f"[RAG] vectorstore has {count} chunks")
    except Exception as e:
        print(f"[RAG] Error checking count: {e}")
        raise RuntimeError(f"Could not read vectorstore: {e}") from e

    if count == 0:
        print("[RAG] vectorstore is empty — skipping retrieval")
        return result

    print(f"[RAG] Searching {count} chunks for relevant content...")

    # Embed query and search
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
        return result

    print(f"[RAG] Retrieved {len(docs)} chunk(s)")

    # Store raw chunks
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "unknown")
        result['raw_chunks'].append({
            'content': doc.page_content,
            'source': source,
            'index': i + 1
        })
        print(
            f"[RAG] Chunk {i+1}: source={source}, preview={doc.page_content[:100]}...")

    result['count'] = len(docs)
    result['has_context'] = True

    # Format as plain text with clear separators
    text_parts = []
    for chunk in result['raw_chunks']:
        text_parts.append(f"[Document: {chunk['source']}]\n{chunk['content']}")
    result['formatted_text'] = "\n\n---\n\n".join(text_parts)

    # Format with clear XML-style tags (works better for instruction-following models)
    xml_parts = []
    for chunk in result['raw_chunks']:
        xml_parts.append(
            f"<document source='{chunk['source']}'>\n{chunk['content']}\n</document>")
    result['formatted_markdown'] = "\n\n".join(xml_parts)

    return result


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
