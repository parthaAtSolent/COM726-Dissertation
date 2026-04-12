"""
app/core/streaming.py
─────────────────────
Streaming response handling.
"""

from __future__ import annotations

import traceback
import streamlit as st
from app.utils.thread_service import save_message


# ══════════════════════════════════════════════════════════════════════════════
# Message persistence
# ══════════════════════════════════════════════════════════════════════════════

def save_message_to_db(thread_id: str, role: str, content: str) -> bool:
    """Save a message to the database."""
    try:
        save_message(thread_id, role, content)
        return True
    except Exception as e:
        print(f"Failed to save message: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# Note: Streaming is now handled directly in chat.py using the chatbot.stream()
# This file is kept for database utilities only
# ══════════════════════════════════════════════════════════════════════════════
