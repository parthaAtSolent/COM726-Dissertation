"""Thread list rendering components."""

import traceback

import streamlit as st
import llms
from app.utils import get_thread_title
from app.ui.sidebar.thread_actions import switch_thread, save_title


def render_thread_row(thread: dict) -> None:
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
            render_edit_title(thread)
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
                switch_thread(tid)

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


def render_edit_title(thread: dict) -> None:
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
                save_title(tid, new_title)
        with col2:
            if st.button("Cancel", key=f"cancel_{tid}", use_container_width=True):
                st.session_state["editing_thread"] = None
                st.rerun()
