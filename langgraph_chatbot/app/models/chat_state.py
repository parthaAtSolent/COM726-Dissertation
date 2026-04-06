"""
app/models/chat_state.py
────────────────────────
Typed state schemas for LangGraph and the service layer.
"""

from __future__ import annotations

from typing import Annotated, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class ChatState(TypedDict):
    """
    Immutable-style state passed between LangGraph nodes.

    messages:     Full conversation history.
    model:        Active model key for this turn.
    routing_info: Metadata about which model was used and why.
    rag_context:  Retrieved document chunks injected into the prompt.
    """

    messages:     Annotated[list[BaseMessage], add_messages]
    model:        str
    routing_info: Optional[dict]
    rag_context:  Optional[str]


class RoutingInfo(TypedDict):
    """Structured routing metadata attached to each AI response."""

    model_key:     str
    model_name:    str
    reason:        str
    auto_selected: bool


class ThreadMeta(TypedDict):
    """In-memory thread record shape."""

    thread_id:  str
    title:      str
    model:      str
    created_at: str
