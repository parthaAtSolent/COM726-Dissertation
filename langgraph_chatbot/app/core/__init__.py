"""
Core Module
===========
Core application logic including the LangGraph workflow and chatbot management.

This module provides the main conversation graph and chatbot singleton
that powers the entire application.
"""

from .graph import chatbot
from app.core.bootstrap import bootstrap_application
from app.core.streaming import process_pending_message, save_message_to_db

__all__ = [
    "chatbot",
    "bootstrap_application",
    "process_pending_message",
    "save_message_to_db",
]
