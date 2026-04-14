"""
UI Module
=========
Streamlit UI components for the chatbot application.

This module provides UI rendering functions for:
- Chat interface (header, history)
- Sidebar navigation
- Model selection
- RAG document management
"""

from app.ui.chat import render_header, render_history, render_chat_page
from app.ui.sidebar import render_sidebar

__all__ = [
    "render_sidebar",
    "render_header",
    "render_history",
    "render_chat_page",
]
