"""
Sidebar module - Navigation and controls.
"""

from app.ui.sidebar.branding import render_branding
from app.ui.sidebar.model_selector import render_model_selector
from app.ui.sidebar.thread_actions import (
    new_chat,
    switch_thread,
    delete_thread,
    save_title,
    on_model_change,
    clear_thread_cache,
    get_cached_threads,
)
from app.ui.sidebar.thread_list import render_thread_row
from app.ui.sidebar.rag_panel import render_rag_panel
from app.ui.sidebar.delete_confirmation import render_delete_confirmation

__all__ = [
    "render_branding",
    "render_model_selector",
    "new_chat",
    "switch_thread",
    "delete_thread",
    "save_title",
    "on_model_change",
    "clear_thread_cache",
    "get_cached_threads",
    "render_thread_row",
    "render_rag_panel",
    "render_delete_confirmation",
    "render_sidebar",
]


def render_sidebar() -> None:
    """Render the complete sidebar."""
    import streamlit as st
    import traceback

    try:
        render_branding()
        render_model_selector()

        if st.sidebar.button("➕ New Chat", use_container_width=True, type="primary"):
            clear_thread_cache()
            new_chat()

        st.sidebar.divider()
        st.sidebar.header("📝 Conversations")

        threads = get_cached_threads()

        print(f"[Sidebar] Rendering {len(threads)} threads")

        if not threads:
            st.sidebar.info("No conversations yet. Start a new chat!")
        else:
            for thread in threads:
                render_thread_row(thread)

        render_delete_confirmation()
        render_rag_panel()

    except Exception as e:
        st.sidebar.error(f"Failed to render sidebar: {str(e)}")
        print(f"Render sidebar error: {traceback.format_exc()}")
