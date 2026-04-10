from __future__ import annotations
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
    """Render the branding header."""
    try:
        st.markdown("""
        <style>
        .branding-bar {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.2rem 0 0.6rem;
            border-bottom: 1px solid var(--border, rgba(108,99,255,0.15));
            margin-bottom: 0.5rem;
        }
        .branding-left {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .branding-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: var(--text-primary, #e0e0f0);
            line-height: 1.2;
        }
        .branding-subtitle {
            font-size: 0.72rem;
            color: var(--text-muted, #7a75a8);
            letter-spacing: 0.06em;
        }
        </style>
        """, unsafe_allow_html=True)

        st.sidebar.markdown("""
        <div class="branding-bar">
            <div class="branding-left">
                <div>
                    <div class="branding-title">LangGraph Chat</div>
                    <div class="branding-subtitle">COM726 · DISSERTATION</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

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
# RAG panel
# ══════════════════════════════════════════════════════════════════════════════

def _render_rag_panel() -> None:
    """Render the document upload panel for RAG."""
    try:
        st.sidebar.divider()
        st.sidebar.header("📄 Documents (RAG)")

        ingested = list_ingested_files()
        if ingested:
            st.sidebar.caption(f"✅ {len(ingested)} file(s) loaded:")
            for fname in ingested:
                st.sidebar.caption(f"  • {fname}")
        else:
            st.sidebar.caption("No documents loaded yet.")

        uploaded_files = st.sidebar.file_uploader(
            "Upload PDF or TXT",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            key="rag_uploader",
            label_visibility="collapsed",
        )

        if uploaded_files:
            if st.sidebar.button(
                "⚙️ Process Documents",
                key="rag_process",
                use_container_width=True,
            ):
                with st.sidebar:
                    with st.spinner("Embedding documents..."):
                        temp_paths = []
                        with tempfile.TemporaryDirectory() as tmp_dir:
                            for uf in uploaded_files:
                                tmp_path = os.path.join(tmp_dir, uf.name)
                                with open(tmp_path, "wb") as f:
                                    f.write(uf.getbuffer())
                                temp_paths.append(tmp_path)

                            chunks = ingest_documents(temp_paths)

                    if chunks > 0:
                        st.success(f"✅ {chunks} chunks stored!")
                        st.rerun()
                    else:
                        st.error("Failed to process documents.")

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
