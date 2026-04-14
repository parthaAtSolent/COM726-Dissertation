"""Chat page UI components."""

from app.ui.chat.message_handler import render_chat_page
import streamlit as st
from app.ui.chat import render_chat_page


def render_header() -> None:
    """Render the chat header."""
    from app.ui.chat.header import render_header as _render_header
    _render_header()


def render_history() -> None:
    """Render the chat history."""
    from app.ui.chat.history import render_history as _render_history
    _render_history()


# Re-export the main chat page renderer

__all__ = ["render_header", "render_history", "render_chat_page"]
