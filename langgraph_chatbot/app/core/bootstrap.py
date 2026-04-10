"""
app/core/bootstrap.py
─────────────────────
Application initialization and bootstrap logic.
"""

from __future__ import annotations

import streamlit as st
import llms
from app.utils.thread_service import (
    create_thread,
    list_threads,
    load_conversation,
    new_thread_id,
)


def bootstrap_application() -> None:
    """Bootstrap the application state."""
    if st.session_state.get("_initialized"):
        return

    try:
        model_keys = llms.list_model_keys()
        default_model = model_keys[0] if model_keys else "llama-8b-instant"
        existing_threads = list_threads()

        if not existing_threads:
            thread_id = new_thread_id()
            create_thread(thread_id, title="New Chat", model=default_model)
            st.session_state["thread_id"] = thread_id
            st.session_state["message_history"] = []
        else:
            first_thread = existing_threads[0]
            thread_id = first_thread["thread_id"]
            default_model = first_thread.get("model", default_model)
            st.session_state["thread_id"] = thread_id
            st.session_state["message_history"] = load_conversation(thread_id)

        st.session_state.update({
            "selected_model": default_model,
            "editing_thread": None,
            "delete_confirm": None,
            "processing_message": False,
            "pending_user_input": None,
            "_initialized": True,
        })

    except Exception as e:
        st.error(f"Failed to initialize application: {str(e)}")
        st.stop()
