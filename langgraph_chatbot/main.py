"""
app/main.py
───────────
Main application entry point.
"""

from __future__ import annotations
from app.utils.thread_service import init_threads_table
from app.utils import inject_css, inject_js
from app.ui import render_sidebar
from app.core import bootstrap_application
from app.ui.chat import render_chat_page
import streamlit as st

import sys
from pathlib import Path

# ── Path configuration ────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ── Page configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LangGraph Chatbot · COM726",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Static assets ─────────────────────────────────────────────────────────────
inject_css("sidebar.css", "chat.css")
inject_js("theme.js", "utils.js", "chat.js")


# ── Database initialization ───────────────────────────────────────────────────
init_threads_table()


# ── Main application flow ─────────────────────────────────────────────────────
def main() -> None:
    """Main application entry point."""
    bootstrap_application()
    render_sidebar()
    render_chat_page()


if __name__ == "__main__":
    main()
