"""
app/main.py
───────────
Main application entry point.
Only function calls, no definitions.
"""

from __future__ import annotations
from app.utils.thread_service import init_threads_table
from app.utils import inject_css, inject_js
from app.ui import render_sidebar, render_header, render_history
from app.core import bootstrap_application, process_pending_message, save_message_to_db
import streamlit as st

import sys
from pathlib import Path

# ── Path configuration ────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Imports ───────────────────────────────────────────────────────────────────


# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LangGraph Chatbot · COM726",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Static assets ─────────────────────────────────────────────────────────────
inject_css("sidebar.css", "chat.css")
inject_js("theme.js", "utils.js")


# ── Database initialization ───────────────────────────────────────────────────
init_threads_table()


# ── Main application flow ─────────────────────────────────────────────────────
def main() -> None:
    """Main application entry point."""

    # Initialize application
    bootstrap_application()

    # Render UI components
    render_sidebar()
    render_header()
    render_history()

    # Handle user input
    user_input = st.chat_input("Type your message here…")

    if user_input and not st.session_state.get("processing_message", False):
        st.session_state.processing_message = True
        st.session_state.message_history.append(
            {"role": "user", "content": user_input})
        save_message_to_db(st.session_state.thread_id, "user", user_input)
        st.session_state.pending_user_input = user_input
        st.rerun()

    # Process pending assistant response
    if st.session_state.get("processing_message", False) and st.session_state.get("pending_user_input"):
        process_pending_message()
        st.rerun()


# ── Application entry point ───────────────────────────────────────────────────
if __name__ == "__main__":
    main()
