"""
Core Module
===========
Core application logic including the LangGraph workflow and chatbot management.

This module provides the main conversation graph and chatbot singleton
that powers the entire application.
"""

from .graph import chatbot

__all__ = [
    "chatbot",
]
