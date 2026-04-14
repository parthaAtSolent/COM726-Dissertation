from __future__ import annotations
import base64
import os
import tempfile
import traceback

import streamlit as st
import llms

from app.utils import (
    create_thread,
    delete_thread,
    get_most_recent_thread_id,
    get_thread_model,
    get_thread_title,
    list_threads,
    load_conversation,
    new_thread_id,
    update_model,
    update_title,
)
from app.core.graph import chatbot
from app.rag import ingest_documents, clear_vectorstore, list_ingested_files


# ══════════════════════════════════════════════════════════════════════════════
# Thread actions
# ══════════════════════════════════════════════════════════════════════════════

def _new_chat() -> None:
    """Create a new chat thread."""
    try:
        tid = new_thread_id()
        model = st.session_state.get(
            "selected_model",
            llms.list_model_keys()[0] if llms.list_model_keys(
            ) else "llama-8b-instant"
        )
        create_thread(tid, title="New Chat", model=model)
        st.session_state.update({
            "thread_id": tid,
            "message_history": [],
            "editing_thread": None,
            "delete_confirm": None,
        })
        _clear_thread_cache()
        st.rerun()
    except Exception as e:
        st.error(f"Failed to create new chat: {str(e)}")
        print(f"New chat error: {traceback.format_exc()}")


def _switch_thread(thread_id: str) -> None:
    """Switch to a different thread."""
    if st.session_state.get("thread_id") == thread_id:
        return

    try:
        print(f"[Sidebar] Switching to thread: {thread_id}")

        # Clear UI states first
        st.session_state["editing_thread"] = None
        st.session_state["delete_confirm"] = None

        # Load the thread data from MySQL
        messages = load_conversation(thread_id)
        model = get_thread_model(thread_id)

        print(
            f"[Sidebar] Loaded {len(messages)} messages for thread {thread_id}")

        # Update session state
        st.session_state.update({
            "thread_id": thread_id,
            "message_history": messages,
            "selected_model": model,
        })

        st.rerun()
    except Exception as e:
        st.error(f"Failed to switch thread: {str(e)}")
        print(f"Switch thread error: {traceback.format_exc()}")


def _delete_thread(thread_id: str) -> None:
    """Delete a thread and its associated data."""
    try:
        print(f"[Sidebar] Deleting thread: {thread_id}")
        success = delete_thread(thread_id)
        if success:
            # Delete from checkpointer if exists
            try:
                if hasattr(chatbot, 'checkpointer') and hasattr(chatbot.checkpointer, 'delete_thread'):
                    chatbot.checkpointer.delete_thread(thread_id)
            except Exception as e:
                print(
                    f"Warning: Could not delete checkpoints for {thread_id}: {e}")

            # Handle current thread deletion
            if st.session_state.get("thread_id") == thread_id:
                recent = get_most_recent_thread_id()
                if recent and recent != thread_id:
                    st.session_state.update({
                        "thread_id": recent,
                        "message_history": load_conversation(recent),
                        "selected_model": get_thread_model(recent),
                    })
                else:
                    _new_chat()
                    return

        st.session_state["delete_confirm"] = None
        _clear_thread_cache()
        st.rerun()
    except Exception as e:
        st.error(f"Failed to delete thread: {str(e)}")
        print(f"Delete thread error: {traceback.format_exc()}")


def _save_title(thread_id: str, new_title: str) -> None:
    """Save edited thread title."""
    try:
        if new_title and new_title.strip():
            success = update_title(thread_id, new_title.strip())
            if success:
                st.toast(f"Title updated to: {new_title.strip()}", icon="✅")
            else:
                st.warning("Failed to update title")
        st.session_state["editing_thread"] = None
        _clear_thread_cache()
        st.rerun()
    except Exception as e:
        st.error(f"Failed to save title: {str(e)}")
        print(f"Save title error: {traceback.format_exc()}")


def _on_model_change() -> None:
    """Handle model selection change."""
    try:
        chosen = st.session_state["model_selector_widget"]
        st.session_state["selected_model"] = chosen
        tid = st.session_state.get("thread_id")
        if tid:
            update_model(tid, chosen)
    except Exception as e:
        st.error(f"Failed to change model: {str(e)}")
        print(f"Model change error: {traceback.format_exc()}")


# ══════════════════════════════════════════════════════════════════════════════
# Thread list caching
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=1)
def _get_cached_threads() -> list:
    """Get cached list of threads from MySQL."""
    try:
        threads = list_threads()
        print(f"[Sidebar] Cached {len(threads)} threads from MySQL")
        return threads
    except Exception as e:
        print(f"[Sidebar] Failed to fetch threads: {e}")
        return []


def _clear_thread_cache() -> None:
    """Clear the thread list cache."""
    print("[Sidebar] Clearing thread cache")
    _get_cached_threads.clear()


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar render functions
# ══════════════════════════════════════════════════════════════════════════════


def _render_branding() -> None:
    """Render the branding header with a Python-driven theme toggle."""
    try:
        # ── 1. Initialise theme state ─────────────────────────────────────────
        if "theme" not in st.session_state:
            st.session_state["theme"] = "dark"

        is_light = st.session_state["theme"] == "light"

        # ── 2. Inject CSS variables inline (guaranteed to apply, Rule 3) ──────
        if is_light:
            theme_css = """
            <style>
            :root {
                --accent:              #6C63FF;
                --accent-light:        #5a52e0;
                --border:              rgba(108, 99, 255, 0.22);
                --border-hover:        rgba(108, 99, 255, 0.50);
                --text-primary:        #1A1A2E;
                --text-muted:          #6b6897;
                --thread-bg:           rgba(108, 99, 255, 0.06);
                --thread-border:       rgba(108, 99, 255, 0.20);
                --thread-hover-bg:     rgba(108, 99, 255, 0.14);
                --thread-hover-border: rgba(108, 99, 255, 0.40);
                --text-thread:         #3a3660;
            }
            
            /* ════════════════════════════════════════════════
               LIGHT MODE — full surface + component overrides
               ════════════════════════════════════════════════ */

            /* ── Page surfaces ── */
            .stApp                                      { background-color: #F5F5FF !important; color: #1A1A2E !important; }
            section[data-testid="stSidebar"]            { background-color: #eeeeff !important; }

            /* ── Top toolbar (Deploy bar) ── */
            header[data-testid="stHeader"],
            [data-testid="stToolbar"],
            .stAppToolbar                               { background-color: #eeeeff !important; border-bottom: 1px solid rgba(108,99,255,0.15) !important; }
            header[data-testid="stHeader"] *,
            [data-testid="stToolbar"] *                 { color: #1A1A2E !important; }

            /* ── All text in the app ── */
            .stApp p, .stApp span, .stApp div,
            .stApp h1, .stApp h2, .stApp h3,
            .stApp h4, .stApp h5, .stApp h6,
            .stApp label, .stApp li, .stApp a,
            [data-testid="stMarkdownContainer"],
            [data-testid="stMarkdownContainer"] *,
            [data-testid="stChatMessage"] *,
            [data-testid="stText"]                      { color: #1A1A2E !important; }

            /* ── Sidebar accent headings stay purple ── */
            section[data-testid="stSidebar"] h2,
            section[data-testid="stSidebar"] h3,
            section[data-testid="stSidebar"] .stSubheader { color: #6C63FF !important; }

            /* ── Sidebar captions and text ── */
            section[data-testid="stSidebar"] .stCaption,
            section[data-testid="stSidebar"] .stMarkdown,
            section[data-testid="stSidebar"] p,
            section[data-testid="stSidebar"] span,
            section[data-testid="stSidebar"] div:not(.branding-bar) { color: #1A1A2E !important; }

            /* ── Model dropdown (select box) ── */
            [data-baseweb="select"],
            [data-baseweb="select"] > div,
            [data-baseweb="select"] input               { background-color: #ffffff !important; color: #1A1A2E !important; border-color: rgba(108,99,255,0.3) !important; }
            [data-baseweb="popover"] *,
            [data-baseweb="menu"] *                     { background-color: #ffffff !important; color: #1A1A2E !important; }

            /* ── File uploader dropzone — IMPROVED LIGHT MODE FIX ── */
            [data-testid="stFileUploadDropzone"] {
                background-color: #ffffff !important;
                background: #ffffff !important;
                border: 2px dashed rgba(108, 99, 255, 0.3) !important;
                border-radius: 8px !important;
                padding: 20px !important;
            }
            
            [data-testid="stFileUploadDropzone"] > div {
                background-color: #ffffff !important;
                background: #ffffff !important;
            }
            
            [data-testid="stFileUploadDropzone"] * {
                background-color: transparent !important;
                color: #1A1A2E !important;
                opacity: 1 !important;
            }
            
            /* Dropzone text and icons */
            [data-testid="stFileUploadDropzone"] svg {
                fill: #6C63FF !important;
                stroke: #6C63FF !important;
            }
            
            [data-testid="stFileUploadDropzone"] p,
            [data-testid="stFileUploadDropzone"] span {
                color: #6b6897 !important;
            }
            
            /* Browse files button */
            [data-testid="stFileUploadDropzone"] button {
                background-color: #e8e7ff !important;
                color: #6C63FF !important;
                border: 1px solid rgba(108, 99, 255, 0.3) !important;
                border-radius: 6px !important;
            }
            
            [data-testid="stFileUploadDropzone"] button:hover {
                background-color: #d8d6ff !important;
            }

            /* ── Chat input bar — white card on light page ── */
            [data-testid="stBottom"],
            [data-testid="stBottom"] > div              { background-color: #F5F5FF !important; }
            [data-testid="stChatInput"],
            [data-testid="stChatInput"] > div           { background-color: #ffffff !important;
                                                          border: 1px solid rgba(108,99,255,0.3) !important;
                                                          border-radius: 12px !important; }
            [data-testid="stChatInput"] textarea        { background-color: #ffffff !important;
                                                          color: #1A1A2E !important; }
            [data-testid="stChatInput"] textarea::placeholder { color: #6b6897 !important; }
            [data-testid="stChatInput"] button          { color: #6C63FF !important; }

            /* ── Info and warning boxes in sidebar ── */
            section[data-testid="stSidebar"] .stAlert,
            section[data-testid="stSidebar"] .stInfo,
            section[data-testid="stSidebar"] .stSuccess,
            section[data-testid="stSidebar"] .stWarning {
                background-color: #e8e7ff !important;
                color: #1A1A2E !important;
                border-left-color: #6C63FF !important;
            }
            
            section[data-testid="stSidebar"] .stAlert p,
            section[data-testid="stSidebar"] .stInfo p,
            section[data-testid="stSidebar"] .stSuccess p {
                color: #1A1A2E !important;
            }

            /* ── Buttons in sidebar ── */
            section[data-testid="stSidebar"] .stButton > button {
                background-color: #e8e7ff !important;
                color: #1A1A2E !important;
                border: 1px solid rgba(108, 99, 255, 0.25) !important;
            }
            
            section[data-testid="stSidebar"] .stButton > button:hover {
                background-color: #d8d6ff !important;
            }
            
            section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
                background-color: #6C63FF !important;
                color: white !important;
            }

            /* ── Icon buttons (edit ✏️ / delete 🗑️) ── */
            section[data-testid="stSidebar"] .stButton > button[kind="secondary"] {
                background-color: #e8e7ff !important;
                border-color: rgba(108,99,255,0.25) !important;
                color: #1A1A2E !important;
            }
            section[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
                background-color: #d8d6ff !important;
            }

            /* ── Thread list buttons (inactive) ── */
            section[data-testid="stSidebar"] .stButton > button[kind="secondary"]:not([data-active]) {
                background-color: rgba(108,99,255,0.07) !important;
                color: #3a3660 !important;
            }
            
            /* ── Divider styling ── */
            section[data-testid="stSidebar"] hr {
                border-color: rgba(108, 99, 255, 0.2) !important;
            }
            </style>
            """
        else:
            theme_css = """
            <style>
            :root {
                --accent:              #6C63FF;
                --accent-light:        #9f9aff;
                --border:              rgba(108, 99, 255, 0.15);
                --border-hover:        rgba(108, 99, 255, 0.40);
                --text-primary:        #e0e0f0;
                --text-muted:          #7a75a8;
                --thread-bg:           rgba(255, 255, 255, 0.04);
                --thread-border:       rgba(108, 99, 255, 0.20);
                --thread-hover-bg:     rgba(108, 99, 255, 0.12);
                --thread-hover-border: rgba(108, 99, 255, 0.45);
                --text-thread:         #b8b4e8;
            }
            
            /* Dark mode file uploader styling */
            [data-testid="stFileUploadDropzone"] {
                background-color: rgba(255, 255, 255, 0.05) !important;
                border: 2px dashed rgba(108, 99, 255, 0.3) !important;
                border-radius: 8px !important;
            }
            
            [data-testid="stFileUploadDropzone"] * {
                color: #e0e0f0 !important;
            }
            
            [data-testid="stFileUploadDropzone"] button {
                background-color: rgba(108, 99, 255, 0.2) !important;
                color: #c4c0ff !important;
            }
            </style>
            """

        # Inject theme vars globally
        st.markdown(theme_css, unsafe_allow_html=True)

        # ── 3. Branding HTML — purely visual, zero interactive JS ─────────────
        pill_class = "theme-switch is-light" if is_light else "theme-switch"
        st.sidebar.markdown(f"""
        <div class="branding-bar">
            <div class="branding-left">
                <span style="font-size:1.6rem;">🧠</span>
                <div>
                    <div class="branding-title">LangGraph Chat</div>
                    <div class="branding-subtitle">COM726 · DISSERTATION</div>
                </div>
            </div>
            <div class="{pill_class}" aria-hidden="true">
                <span class="switch-moon">🌙</span>
                <span class="switch-thumb"></span>
                <span class="switch-sun">☀️</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── 4. Actual toggle — native Streamlit button ────────────────────────
        toggle_label = "Switch to 🌙 Dark" if is_light else "Switch to ☀️ Light"
        if st.sidebar.button(toggle_label, key="theme_toggle_btn",
                             use_container_width=True):
            st.session_state["theme"] = "dark" if is_light else "light"
            st.rerun()

    except Exception as e:
        st.sidebar.error(f"Failed to load branding: {str(e)}")


def _render_model_selector() -> None:
    """Render the model selector dropdown."""
    try:
        keys = llms.list_model_keys()
        if not keys:
            st.sidebar.error("No models available")
            return

        display_map = {k: llms.get_display_name(k) for k in keys}
        current = st.session_state.get(
            "selected_model", keys[0] if keys else None)
        idx = keys.index(current) if current in keys else 0

        st.sidebar.subheader("🤖 Model")

        st.sidebar.selectbox(
            "Choose AI Model",
            options=keys,
            format_func=lambda k: display_map.get(k, k),
            index=idx,
            key="model_selector_widget",
            on_change=_on_model_change,
            help="Select which AI model to use for this conversation"
        )
        st.sidebar.caption("⚡ Powered by Groq's free tier")
        st.sidebar.divider()
    except Exception as e:
        st.sidebar.error(f"Failed to load model selector: {str(e)}")


def _render_delete_confirmation() -> None:
    """Render delete confirmation dialog."""
    thread_id = st.session_state.get("delete_confirm")
    if not thread_id:
        return

    thread_title = get_thread_title(thread_id)
    with st.sidebar:
        st.markdown("### ⚠️ Delete Conversation")
        st.markdown(f"Are you sure you want to delete **{thread_title}**?")
        st.markdown("This action cannot be undone.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Yes, Delete", key="confirm_delete", use_container_width=True):
                _delete_thread(thread_id)
        with col2:
            if st.button("Cancel", key="cancel_delete", use_container_width=True):
                st.session_state["delete_confirm"] = None
                st.rerun()
        st.divider()


def _render_edit_title(thread: dict) -> None:
    """Render inline title editor."""
    tid = thread["thread_id"]
    current_title = thread.get("title", "Untitled")
    with st.sidebar:
        st.markdown("---")
        st.markdown("**✏️ Edit Title**")
        new_title = st.text_input(
            "Title",
            value=current_title,
            key=f"edit_input_{tid}",
            label_visibility="collapsed"
        )
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save", key=f"save_{tid}", use_container_width=True):
                _save_title(tid, new_title)
        with col2:
            if st.button("Cancel", key=f"cancel_{tid}", use_container_width=True):
                st.session_state["editing_thread"] = None
                st.rerun()


def _render_thread_row(thread: dict) -> None:
    """Render a single thread row in the sidebar."""
    try:
        tid = thread["thread_id"]
        title = thread.get("title", "Untitled")
        model_keys = llms.list_model_keys()
        model_key = thread.get(
            "model", model_keys[0] if model_keys else "llama-8b-instant")
        icon = llms.get_icon(model_key)
        is_active = st.session_state.get("thread_id") == tid

        if st.session_state.get("editing_thread") == tid:
            _render_edit_title(thread)
            return

        label = f"{'📌 ' if is_active else ''}{icon} {title[:40]}{'...' if len(title) > 40 else ''}"
        col1, col2, col3 = st.sidebar.columns([6, 1, 1])

        with col1:
            if st.button(
                label,
                key=f"t_{tid}",
                use_container_width=True,
                type="primary" if is_active else "secondary"
            ):
                _switch_thread(tid)

        with col2:
            if st.button("✏️", key=f"e_{tid}", help="Edit title"):
                st.session_state["editing_thread"] = tid
                st.session_state["delete_confirm"] = None
                st.rerun()

        with col3:
            if st.button("🗑️", key=f"d_{tid}", help="Delete conversation"):
                st.session_state["delete_confirm"] = tid
                st.session_state["editing_thread"] = None
                st.rerun()

    except Exception as e:
        st.sidebar.error(f"Failed to render thread: {str(e)}")
        print(f"Render thread error: {traceback.format_exc()}")


# ══════════════════════════════════════════════════════════════════════════════
# RAG panel - FIXED VERSION
# ══════════════════════════════════════════════════════════════════════════════

def _render_rag_panel() -> None:
    """Render the document upload panel for RAG using the custom file uploader."""
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

        # ── 2. File uploader (Custom Styling Applied)───────────────────────────────────────────

        st.markdown("""
        <style>
        /* Scope only to the sidebar uploader */
        section[data-testid="stSidebar"] div.st-key-rag_uploader_native [data-testid="stFileUploaderDropzone"] {
            background: linear-gradient(135deg, #6C63FF, #5a52e0) !important;
            border: 3px solid #4a43c0 !important;
            border-radius: 16px !important;
            padding: 1rem !important;
            box-shadow: none !important;
        }

        /* Inner layers */
        section[data-testid="stSidebar"] div.st-key-rag_uploader_native [data-testid="stFileUploaderDropzone"] > div,
        section[data-testid="stSidebar"] div.st-key-rag_uploader_native [data-testid="stFileUploaderDropzone"] > div > div {
            background: transparent !important;
        }

        /* Text */
        section[data-testid="stSidebar"] div.st-key-rag_uploader_native [data-testid="stFileUploaderDropzone"] * {
            color: #ffffff !important;
            opacity: 1 !important;
            font-weight: 600 !important;
        }

        /* Icon */
        section[data-testid="stSidebar"] div.st-key-rag_uploader_native [data-testid="stFileUploaderDropzone"] svg {
            fill: #ffffff !important;
        }

        /* Button */
        section[data-testid="stSidebar"] div.st-key-rag_uploader_native [data-testid="stFileUploaderDropzone"] button {
            background: #ffffff !important;
            color: #5a52e0 !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 0.45rem 0.9rem !important;
            font-weight: 600 !important;
            transition: all 0.2s ease !important;
        }
        
        /* Button hover effect */
        section[data-testid="stSidebar"] div.st-key-rag_uploader_native [data-testid="stFileUploaderDropzone"] button:hover {
            background: #f0efff !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15) !important;
        }
        </style>
        """, unsafe_allow_html=True)

        files = st.sidebar.file_uploader(
            "Upload PDF or TXT files",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            key="rag_uploader_native",
        )

        # ── 3. Process button with detailed debugging ────────────────────
        if files:
            n = len(files)
            label = f"⚙️ Process {n} file{'s' if n > 1 else ''}"

            # Debug: Show what files were selected
            st.sidebar.write(
                f"📁 Selected files: {[f.name for f in files]}")

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

                                    # Handle UploadedFile objects (native Streamlit)
                                    tmp_path = os.path.join(tmp_dir, f.name)
                                    with open(tmp_path, "wb") as fh:
                                        fh.write(f.getvalue())
                                    temp_paths.append(tmp_path)
                                    status_text.success(
                                        f"  ✅ Loaded: {f.name} ({f.size} bytes)")

                                if not temp_paths:
                                    status_text.error(
                                        "No valid files to process")
                                    return

                                status_text.info(
                                    f"📥 Ingesting {len(temp_paths)} file(s)...")
                                progress.caption("🧠 Embedding chunks…")

                                # This is the critical call - where ingestion happens
                                status_text.info(
                                    "Calling ingest_documents()...")
                                chunks = ingest_documents(temp_paths)
                                status_text.success(
                                    f"✅ ingest_documents returned {chunks} chunks")

                                # Verify immediately
                                new_count = get_vectorstore_count()
                                status_text.info(
                                    f"📊 Vectorstore now has {new_count} chunks")

                    if chunks > 0:
                        st.sidebar.success(
                            f"✅ {chunks} chunks stored from {n} file(s)"
                        )
                        st.rerun()
                    else:
                        st.sidebar.warning(
                            "⚠️ Files were loaded but produced no content. "
                            "Check console for errors."
                        )

                except RuntimeError as e:
                    st.sidebar.error(f"❌ {e}")
                    st.sidebar.info(
                        "💡 Make sure Ollama is running:\n"
                        "```bash\n"
                        "ollama serve\n"
                        "ollama pull nomic-embed-text\n"
                        "```"
                    )
                except Exception as e:
                    st.sidebar.error(f"❌ Unexpected error: {e}")
                    st.sidebar.code(str(e))
                    print(f"RAG panel error: {traceback.format_exc()}")
                finally:
                    progress.empty()
                    status_text.empty()

        # ── 4. Clear all documents ────────────────────────────────────────────
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


# ══════════════════════════════════════════════════════════════════════════════
# Public entry point
# ══════════════════════════════════════════════════════════════════════════════


def render_sidebar() -> None:
    """Render the complete sidebar."""
    try:
        _render_branding()
        _render_model_selector()

        if st.sidebar.button("➕ New Chat", use_container_width=True, type="primary"):
            _clear_thread_cache()
            _new_chat()

        st.sidebar.divider()
        st.sidebar.header("📝 Conversations")

        threads = _get_cached_threads()

        print(f"[Sidebar] Rendering {len(threads)} threads")

        if not threads:
            st.sidebar.info("No conversations yet. Start a new chat!")
        else:
            for thread in threads:
                _render_thread_row(thread)

        _render_delete_confirmation()
        _render_rag_panel()

    except Exception as e:
        st.sidebar.error(f"Failed to render sidebar: {str(e)}")
        print(f"Render sidebar error: {traceback.format_exc()}")
