"""
LangGraph Chatbot Application
=============================
A professional chatbot built with LangGraph and Streamlit.

This package contains the core application logic, including:
- Graph-based conversation workflow
- LLM model management
- Thread and conversation persistence
- Web interface components
"""

from .models.chat_state import ChatState, RoutingInfo, ThreadMeta
import sys
from pathlib import Path

# Ensure COM726 - Dissertation root is on sys.path so `llms` and `config` resolve
# app/ -> langgraph_chatbot/ -> COM726 - Dissertation/
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

__version__ = "1.0.0"
__author__ = "COM726 Project"


__all__ = [
    'ChatState',
    'RoutingInfo',
    'ThreadMeta',
]
