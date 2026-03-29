"""
app/utils/thread_service.py
────────────────────────────
Pure in-memory thread management.  No SQL, no Streamlit imports.

ACID notes
──────────
Atomicity  — each mutation is a single dict operation.
Consistency — thread IDs are validated as UUID v4 before storage.
Isolation   — Streamlit's single-threaded model keeps this safe.
Durability  — message content is persisted by LangGraph's SqliteSaver;
              titles live in memory and are regenerated on restart.
"""

from __future__ import annotations
from config.settings import (
    DEFAULT_MODEL_KEY,
    DEFAULT_THREAD_TITLE,
    MAX_TITLE_LENGTH,
    TITLE_PROMPT_MAX_CHARS,
)
import llms

import uuid
from typing import Dict, List, Optional

from langchain_core.messages import HumanMessage

from app.models.chat_state import ThreadMeta

# # llms is at COM726/ root
# import sys
# from pathlib import Path
# # langgraph_chatbot/
# sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


# ── In-memory store ────────────────────────────────────────────────────────────
_threads: Dict[str, ThreadMeta] = {}


# ── Helpers ────────────────────────────────────────────────────────────────────

def new_thread_id() -> str:
    return str(uuid.uuid4())


def _validate(thread_id: str) -> None:
    try:
        uuid.UUID(thread_id)
    except ValueError as exc:
        raise ValueError(f"Invalid thread_id: '{thread_id}'") from exc


# ── CRUD ───────────────────────────────────────────────────────────────────────

def create_thread(
    thread_id: str,
    title: str = DEFAULT_THREAD_TITLE,
    model: str = DEFAULT_MODEL_KEY,
    created_at: str = "",
) -> ThreadMeta:
    _validate(thread_id)
    meta: ThreadMeta = {
        "thread_id": thread_id,
        "title": title,
        "model": model,
        "created_at": created_at,
    }
    _threads[thread_id] = meta
    return meta


def get_thread(thread_id: str) -> Optional[ThreadMeta]:
    return _threads.get(thread_id)


def update_title(thread_id: str, new_title: str) -> bool:
    if thread_id not in _threads:
        return False
    _threads[thread_id]["title"] = new_title[:MAX_TITLE_LENGTH]
    return True


def update_model(thread_id: str, model_key: str) -> bool:
    if thread_id not in _threads:
        return False
    _threads[thread_id]["model"] = model_key
    return True


def delete_thread(thread_id: str) -> bool:
    return bool(_threads.pop(thread_id, None))


def list_threads() -> List[ThreadMeta]:
    """Return all threads, newest first."""
    return list(reversed(list(_threads.values())))


def get_most_recent_thread_id() -> Optional[str]:
    return next(reversed(_threads), None) if _threads else None


def get_thread_title(thread_id: str) -> str:
    meta = _threads.get(thread_id)
    return meta["title"] if meta else DEFAULT_THREAD_TITLE


def get_thread_model(thread_id: str) -> str:
    meta = _threads.get(thread_id)
    return meta["model"] if meta else DEFAULT_MODEL_KEY


# ── Title generation ───────────────────────────────────────────────────────────

def generate_title(user_prompt: str, model_key: str = DEFAULT_MODEL_KEY) -> str:
    """Ask the LLM to produce a short chat title from the first user message."""
    if not user_prompt.strip():
        return DEFAULT_THREAD_TITLE

    snippet = user_prompt[:TITLE_PROMPT_MAX_CHARS]
    prompt = (
        f"Create a short, descriptive title (max 5–6 words) for a chat "
        f"starting with: '{snippet}'. Reply with ONLY the title."
    )
    try:
        llm = llms.build_llm(model_key)
        raw = llm.invoke([HumanMessage(content=prompt)]).content
        title = raw.strip().strip('"').strip("'")
        return title[:MAX_TITLE_LENGTH] or DEFAULT_THREAD_TITLE
    except Exception as exc:
        print(f"[thread_service] Title generation failed: {exc}")
        return DEFAULT_THREAD_TITLE
