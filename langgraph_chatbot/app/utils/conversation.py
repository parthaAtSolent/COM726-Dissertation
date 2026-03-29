"""
app/utils/conversation.py
──────────────────────────
Pure utility functions for loading and formatting conversation history.

Keeping these separate from the graph and service layers means they can be
tested in isolation without spinning up a LangGraph instance.
"""

from __future__ import annotations

from typing import List

from langchain_core.messages import HumanMessage

from app.core.graph import chatbot


def load_conversation(thread_id: str) -> List[dict]:
    """
    Retrieve formatted message history for *thread_id* from the LangGraph
    checkpointer.

    Returns
    -------
    list of dicts with keys ``role`` (``'user'`` | ``'assistant'``) and
    ``content`` (str).  Returns an empty list on any error.
    """
    try:
        state = chatbot.get_state(
            config={"configurable": {"thread_id": thread_id}}
        )
        raw_messages = state.values.get("messages", [])

        formatted: List[dict] = []
        for msg in raw_messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            formatted.append({"role": role, "content": msg.content})

        return formatted
    except Exception as exc:
        print(f"[conversation] Failed to load thread '{thread_id}': {exc}")
        return []
