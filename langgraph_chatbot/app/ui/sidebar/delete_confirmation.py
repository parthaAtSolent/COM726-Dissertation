"""Delete confirmation dialog component."""

import streamlit as st
from app.utils import get_thread_title
from app.ui.sidebar.thread_actions import delete_thread


def render_delete_confirmation() -> None:
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
                delete_thread(thread_id)
        with col2:
            if st.button("Cancel", key="cancel_delete", use_container_width=True):
                st.session_state["delete_confirm"] = None
                st.rerun()
        st.divider()
