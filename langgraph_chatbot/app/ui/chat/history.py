"""Chat message history component."""

import streamlit as st


def render_history() -> None:
    """Render the chat message history."""
    try:
        for msg in st.session_state.get("message_history", []):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
    except Exception as e:
        st.error(f"Failed to render history: {str(e)}")
