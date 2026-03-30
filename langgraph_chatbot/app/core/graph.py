"""
app/core/graph.py
──────────────────
Defines and compiles the LangGraph StateGraph.

Flow
────
    START ──▶ [ chat_node ] ──▶ END

chat_node
  1. Reads the model key from state.
  2. Delegates LLM instantiation to llms.build_llm() — zero knowledge of
     which model is active; that decision belongs to the llms/ layer.
  3. Builds a plain-text conversation prompt for cross-model compatibility.
  4. Returns updated state including routing_info for the UI to display.

The compiled chatbot is a module-level singleton — built once per Streamlit
worker process and shared across all sessions.
"""

from __future__ import annotations
from config.settings import DATABASE_PATH, DEFAULT_MODEL_KEY
from app.models.chat_state import ChatState, RoutingInfo
import llms

import sqlite3

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph


# ── SQLite connection (LangGraph checkpointing only) ──────────────────────────
_conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
_checkpointer = SqliteSaver(conn=_conn)


# ── Chat node ──────────────────────────────────────────────────────────────────

def chat_node(state: ChatState) -> dict:
    """
    Core LangGraph node.

    Reads conversation state → calls the active LLM → returns state update.
    """
    messages = state["messages"]
    model_key: str = state.get("model") or DEFAULT_MODEL_KEY

    # Extract latest user message
    last_user_message: str | None = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            last_user_message = msg.content
            break

    if not last_user_message:
        return {
            "messages": [AIMessage(content="I didn't receive a message.")],
            "routing_info": None,
        }

    # Build routing metadata for UI display
    routing_info: RoutingInfo = {
        "model_key": model_key,
        "model_name": llms.get_display_name(model_key),
        "reason": "User-selected model.",
        "auto_selected": False,
    }

    # Build plain-text prompt (maximises cross-model compatibility)
    lines = []
    for msg in messages:
        prefix = "User" if isinstance(msg, HumanMessage) else "Assistant"
        lines.append(f"{prefix}: {msg.content}")
    lines.append("Assistant:")
    prompt_text = "\n".join(lines)

    # Invoke LLM via the llms registry
    try:
        llm = llms.build_llm(model_key)
        response = llm.invoke(prompt_text)
        return {
            "messages": [AIMessage(content=response.content)],
            "routing_info": routing_info,
        }
    except Exception as exc:
        print(f"[graph.chat_node] Error: {exc}")
        return {
            "messages": [AIMessage(content=f"⚠️ {exc}")],
            "routing_info": routing_info,
        }


# ── Compiled chatbot singleton ────────────────────────────────────────────────

def _compile() -> object:
    builder = StateGraph(ChatState)
    builder.add_node("chat_node", chat_node)
    builder.add_edge(START, "chat_node")
    builder.add_edge("chat_node", END)
    return builder.compile(checkpointer=_checkpointer)


chatbot = _compile()
