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

from .conversation import (
    load_conversation,
)

from .style_loader import (
    inject_css,
    inject_js,
    load_html_template as load_template,
)

from .thread_service import (
    create_thread,
    delete_thread,
    generate_title,
    get_most_recent_thread_id,
    get_thread_model,
    get_thread_title,
    list_threads,
    new_thread_id,
    update_model,
    update_title,
)

__all__ = [
    # Conversation utilities
    "load_conversation",

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
