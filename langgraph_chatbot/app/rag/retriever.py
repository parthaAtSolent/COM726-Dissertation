"""
app/rag/retriever.py
─────────────────────
Enhanced retrieval with model-compatible formatting.
Now with MMR (Maximum Marginal Relevance) for diverse results.
"""

from __future__ import annotations

from langchain_chroma import Chroma
from app.rag.ingestion import VECTORSTORE_DIR, COLLECTION_NAME
from app.rag.embedder import get_embedder

# 🚀 INCREASED FOR BETTER CONTEXT
TOP_K = 10                    # Was 4 - now retrieves 10 chunks
MMR_FETCH_K = 20             # Fetch 20, diversify to 10 for better coverage
SIMILARITY_THRESHOLD = 0.7   # Minimum similarity score (0-1)


def retrieve_context(query: str, model_key: str = None, use_mmr: bool = True) -> dict:
    """
    Run similarity search and return structured context.

    Args:
        query: User's question
        model_key: Optional model key (reserved for future model-specific formatting)
        use_mmr: Use MMR for diverse results instead of pure similarity

    Returns:
        dict with keys:
        - 'has_context': bool
        - 'raw_chunks': list of dicts with 'content', 'source', 'score'
        - 'formatted_text': str ready for prompt
        - 'formatted_markdown': str with XML tags (better for some models)
        - 'count': int number of chunks
        - 'total_context_length': total characters retrieved
    """
    print(f"[RAG] Retrieving context for query: '{query[:100]}...'")
    print(f"[RAG] vectorstore path: {VECTORSTORE_DIR}")

    result = {
        'has_context': False,
        'raw_chunks': [],
        'formatted_text': '',
        'formatted_markdown': '',
        'count': 0,
        'total_context_length': 0
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

    # Embed query and search with improved method
    try:
        if use_mmr and count >= TOP_K:
            # MMR: Maximum Marginal Relevance - gets diverse relevant chunks
            print(f"[RAG] Using MMR search (fetch={MMR_FETCH_K}, k={TOP_K})")
            docs = vectorstore.max_marginal_relevance_search(
                query,
                k=TOP_K,
                fetch_k=MMR_FETCH_K,
                lambda_mult=0.5  # Balance between relevance and diversity
            )
        else:
            # Regular similarity search
            print(f"[RAG] Using similarity search (k={TOP_K})")
            docs = vectorstore.similarity_search(query, k=TOP_K)

        print(f"[RAG] Search returned {len(docs)} documents")

        # Try to get similarity scores if available
        try:
            docs_with_scores = vectorstore.similarity_search_with_score(
                query, k=TOP_K)
            print(
                f"[RAG] Got similarity scores for {len(docs_with_scores)} docs")
            # Use scores if we have them
            docs = [doc for doc, _ in docs_with_scores]
            scores = [score for _, score in docs_with_scores]
        except:
            scores = [None] * len(docs)
            print(f"[RAG] Similarity scores not available")

    except Exception as e:
        print(f"[RAG] Search failed: {e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(
            f"Similarity search failed — is Ollama running with 'nomic-embed-text'?\n{e}"
        ) from e

    if not docs:
        print("[RAG] No relevant chunks found")
        return result

    print(f"[RAG] Retrieved {len(docs)} chunk(s)")

    # Store raw chunks with metadata
    total_chars = 0
    for i, (doc, score) in enumerate(zip(docs, scores)):
        source = doc.metadata.get("source", "unknown")
        content_length = len(doc.page_content)
        total_chars += content_length

        chunk_info = {
            'content': doc.page_content,
            'source': source,
            'index': i + 1,
            'length': content_length,
        }
        if score is not None:
            chunk_info['similarity_score'] = float(score)

        result['raw_chunks'].append(chunk_info)
        print(
            f"[RAG] Chunk {i+1}: source={source}, length={content_length} chars, score={score if score else 'N/A'}, preview={doc.page_content[:100]}...")

    result['count'] = len(docs)
    result['has_context'] = True
    result['total_context_length'] = total_chars
    print(
        f"[RAG] Total context length: {total_chars} characters ({total_chars//4} tokens approx)")

    # Format as plain text with clear separators and section markers
    text_parts = []
    for i, chunk in enumerate(result['raw_chunks'], 1):
        text_parts.append(f"""
[SOURCE DOCUMENT {i}: {chunk['source']}]
{chunk['content']}
""".strip())
    result['formatted_text'] = "\n\n" + "\n\n---\n\n".join(text_parts)

    # Format with clear XML-style tags (works better for instruction-following models)
    xml_parts = []
    for i, chunk in enumerate(result['raw_chunks'], 1):
        xml_parts.append(f"""
<document id='{i}' source='{chunk['source']}'>
{chunk['content']}
</document>
""".strip())
    result['formatted_markdown'] = "\n\n".join(xml_parts)

    # Also provide a compact version for long contexts
    compact_parts = []
    for chunk in result['raw_chunks']:
        compact_parts.append(
            f"[{chunk['source']}] {chunk['content'][:500]}...")
    result['formatted_compact'] = "\n\n".join(compact_parts)

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
