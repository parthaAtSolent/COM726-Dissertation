"""
app/models/chat_state.py
────────────────────────
Typed state schemas for LangGraph and the service layer.

Dependency direction (strictly enforced):
    models  ←  services  ←  core (graph)  ←  main.py
"""

from __future__ import annotations

from typing import Annotated, Optional, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class ChatState(TypedDict):
    """
    Immutable-style state passed between LangGraph nodes.

    messages:
        Full conversation history.  LangGraph's add_messages reducer
        appends new messages rather than replacing the list — required
        for correct SQLite checkpointing (Durability).
    model:
        Active model key for this turn (matches llms/REGISTRY key).
    routing_info:
        Metadata about which model was used and why; consumed by the UI.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    model: str
    routing_info: Optional[dict]


class RoutingInfo(TypedDict):
    """Structured routing metadata attached to each AI response."""

    model_key: str
    model_name: str
    reason: str
    auto_selected: bool


class ThreadMeta(TypedDict):
    """In-memory thread record shape."""

    thread_id: str
    title: str
    model: str
    created_at: str
