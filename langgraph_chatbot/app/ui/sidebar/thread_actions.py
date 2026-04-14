"""Thread management actions."""

import traceback
import streamlit as st
import llms

from app.utils import (
    create_thread,
    delete_thread as delete_thread_db,
    get_most_recent_thread_id,
    get_thread_model,
    load_conversation,
    new_thread_id,
    update_model,
    update_title,
)
from app.core.graph import chatbot


def new_chat() -> None:
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
        clear_thread_cache()
        st.rerun()
    except Exception as e:
        st.error(f"Failed to create new chat: {str(e)}")
        print(f"New chat error: {traceback.format_exc()}")


def switch_thread(thread_id: str) -> None:
    """Switch to a different thread."""
    if st.session_state.get("thread_id") == thread_id:
        return

    try:
        print(f"[Sidebar] Switching to thread: {thread_id}")

        st.session_state["editing_thread"] = None
        st.session_state["delete_confirm"] = None

        messages = load_conversation(thread_id)
        model = get_thread_model(thread_id)

        print(
            f"[Sidebar] Loaded {len(messages)} messages for thread {thread_id}")

        st.session_state.update({
            "thread_id": thread_id,
            "message_history": messages,
            "selected_model": model,
        })

        st.rerun()
    except Exception as e:
        st.error(f"Failed to switch thread: {str(e)}")
        print(f"Switch thread error: {traceback.format_exc()}")


def delete_thread(thread_id: str) -> None:
    """Delete a thread and its associated data."""
    try:
        print(f"[Sidebar] Deleting thread: {thread_id}")
        success = delete_thread_db(thread_id)
        if success:
            try:
                if hasattr(chatbot, 'checkpointer') and hasattr(chatbot.checkpointer, 'delete_thread'):
                    chatbot.checkpointer.delete_thread(thread_id)
            except Exception as e:
                print(
                    f"Warning: Could not delete checkpoints for {thread_id}: {e}")

            if st.session_state.get("thread_id") == thread_id:
                recent = get_most_recent_thread_id()
                if recent and recent != thread_id:
                    st.session_state.update({
                        "thread_id": recent,
                        "message_history": load_conversation(recent),
                        "selected_model": get_thread_model(recent),
                    })
                else:
                    new_chat()
                    return

        st.session_state["delete_confirm"] = None
        clear_thread_cache()
        st.rerun()
    except Exception as e:
        st.error(f"Failed to delete thread: {str(e)}")
        print(f"Delete thread error: {traceback.format_exc()}")


def save_title(thread_id: str, new_title: str) -> None:
    """Save edited thread title."""
    try:
        if new_title and new_title.strip():
            success = update_title(thread_id, new_title.strip())
            if success:
                st.toast(f"Title updated to: {new_title.strip()}", icon="✅")
            else:
                st.warning("Failed to update title")
        st.session_state["editing_thread"] = None
        clear_thread_cache()
        st.rerun()
    except Exception as e:
        st.error(f"Failed to save title: {str(e)}")
        print(f"Save title error: {traceback.format_exc()}")


def on_model_change() -> None:
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


@st.cache_data(ttl=1)
def get_cached_threads() -> list:
    """Get cached list of threads from MySQL."""
    from app.utils import list_threads
    try:
        threads = list_threads()
        print(f"[Sidebar] Cached {len(threads)} threads from MySQL")
        return threads
    except Exception as e:
        print(f"[Sidebar] Failed to fetch threads: {e}")
        return []


def clear_thread_cache() -> None:
    """Clear the thread list cache."""
    print("[Sidebar] Clearing thread cache")
    get_cached_threads.clear()
