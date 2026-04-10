"""
Utilities Module
================
Helper utilities for the chatbot application.

This module provides various utility functions for:
- Loading and managing conversations
- Injecting CSS/JS styles
- Thread management services
- Common helper functions
"""

# Import thread service functions (MySQL-based - PRIMARY SOURCE)
from .thread_service import (
    create_thread,
    delete_thread,
    generate_title,
    get_most_recent_thread_id,
    get_thread_model,
    get_thread_title,
    list_threads,
    new_thread_id,
    save_message,
    update_model,
    update_title,
)
# Import MySQL load_conversation and alias the checkpointer version
from .thread_service import load_conversation as load_conversation
from .conversation import load_conversation as load_conversation_from_checkpointer

from .style_loader import (
    inject_css,
    inject_js,
    load_html_template as load_template,
)

__all__ = [
    # Conversation utilities - MySQL version is primary
    "load_conversation",
    "load_conversation_from_checkpointer",
    "save_message",

    # Style utilities
    "inject_css",
    "inject_js",
    "load_template",

    # Thread service
    "create_thread",
    "delete_thread",
    "generate_title",
    "get_most_recent_thread_id",
    "get_thread_model",
    "get_thread_title",
    "list_threads",
    "new_thread_id",
    "update_model",
    "update_title",
]
