"""RAG document upload and management panel."""

import os
import tempfile
import traceback
import streamlit as st
from app.rag import ingest_documents, clear_vectorstore, list_ingested_files


def render_rag_panel() -> None:
    """Render the document upload panel for RAG."""
    try:
        from app.rag.ingestion import get_vectorstore_count

        st.sidebar.divider()
        st.sidebar.header("📄 Documents (RAG)")

        # ── 1. Vectorstore status ─────────────────────────────────────────────
        ingested = list_ingested_files()
        chunk_count = get_vectorstore_count()

        if ingested:
            st.sidebar.caption(
                f"✅ {len(ingested)} file(s) · {chunk_count} chunks indexed:"
            )
            for fname in ingested:
                st.sidebar.caption(f"  • {fname}")
        else:
            st.sidebar.caption("No documents loaded yet.")
            st.sidebar.caption(f"📁 Vectorstore path: data/vectorstore")
            st.sidebar.caption(f"📊 Chunks: {chunk_count}")

        # ── 2. File uploader styling ─────────────────────────────────────────
        _inject_uploader_styles()

        files = st.sidebar.file_uploader(
            "Upload PDF or TXT files",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            key="rag_uploader_native",
        )

        # ── 3. Process button ────────────────────────────────────────────────
        if files:
            _process_files(files)

        # ── 4. Clear all documents ───────────────────────────────────────────
        if ingested:
            if st.sidebar.button(
                "🗑️ Clear All Documents",
                key="rag_clear",
                use_container_width=True,
            ):
                clear_vectorstore()
                st.sidebar.success("Documents cleared.")
                st.rerun()

    except Exception as e:
        st.sidebar.error(f"RAG panel error: {str(e)}")
        print(f"RAG panel error: {traceback.format_exc()}")


def _inject_uploader_styles() -> None:
    """Inject custom CSS for the file uploader."""
    st.markdown("""
    <style>
    section[data-testid="stSidebar"] div.st-key-rag_uploader_native [data-testid="stFileUploaderDropzone"] {
        background: linear-gradient(135deg, #6C63FF, #5a52e0) !important;
        border: 3px solid #4a43c0 !important;
        border-radius: 16px !important;
        padding: 1rem !important;
        box-shadow: none !important;
    }
    section[data-testid="stSidebar"] div.st-key-rag_uploader_native [data-testid="stFileUploaderDropzone"] > div,
    section[data-testid="stSidebar"] div.st-key-rag_uploader_native [data-testid="stFileUploaderDropzone"] > div > div {
        background: transparent !important;
    }
    section[data-testid="stSidebar"] div.st-key-rag_uploader_native [data-testid="stFileUploaderDropzone"] * {
        color: #ffffff !important;
        opacity: 1 !important;
        font-weight: 600 !important;
    }
    section[data-testid="stSidebar"] div.st-key-rag_uploader_native [data-testid="stFileUploaderDropzone"] svg {
        fill: #ffffff !important;
    }
    section[data-testid="stSidebar"] div.st-key-rag_uploader_native [data-testid="stFileUploaderDropzone"] button {
        background: #ffffff !important;
        color: #5a52e0 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.45rem 0.9rem !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    section[data-testid="stSidebar"] div.st-key-rag_uploader_native [data-testid="stFileUploaderDropzone"] button:hover {
        background: #f0efff !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15) !important;
    }
    </style>
    """, unsafe_allow_html=True)


def _process_files(files: list) -> None:
    """Process uploaded files and ingest into vectorstore."""
    n = len(files)
    label = f"⚙️ Process {n} file{'s' if n > 1 else ''}"

    st.sidebar.write(f"📁 Selected files: {[f.name for f in files]}")

    if st.sidebar.button(label, key="rag_process", use_container_width=True):
        progress = st.sidebar.empty()
        status_text = st.sidebar.empty()

        try:
            with st.sidebar:
                with st.spinner("Processing documents..."):
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        temp_paths = []

                        for idx, f in enumerate(files):
                            status_text.info(
                                f"📥 Processing file {idx+1}/{n}: {f.name}")
                            tmp_path = os.path.join(tmp_dir, f.name)
                            with open(tmp_path, "wb") as fh:
                                fh.write(f.getvalue())
                            temp_paths.append(tmp_path)
                            status_text.success(
                                f"  ✅ Loaded: {f.name} ({f.size} bytes)")

                        if not temp_paths:
                            status_text.error("No valid files to process")
                            return

                        status_text.info(
                            f"📥 Ingesting {len(temp_paths)} file(s)...")
                        progress.caption("🧠 Embedding chunks…")

                        chunks = ingest_documents(temp_paths)
                        status_text.success(
                            f"✅ ingest_documents returned {chunks} chunks")

                        from app.rag.ingestion import get_vectorstore_count
                        new_count = get_vectorstore_count()
                        status_text.info(
                            f"📊 Vectorstore now has {new_count} chunks")

            if chunks > 0:
                st.sidebar.success(
                    f"✅ {chunks} chunks stored from {n} file(s)")
                st.rerun()
            else:
                st.sidebar.warning(
                    "⚠️ Files were loaded but produced no content. Check console for errors."
                )

        except RuntimeError as e:
            st.sidebar.error(f"❌ {e}")
            st.sidebar.info(
                "💡 Make sure Ollama is running:\n"
                "```bash\nollama serve\nollama pull nomic-embed-text\n```"
            )
        except Exception as e:
            st.sidebar.error(f"❌ Unexpected error: {e}")
            st.sidebar.code(str(e))
            print(f"RAG panel error: {traceback.format_exc()}")
        finally:
            progress.empty()
            status_text.empty()
