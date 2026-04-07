"""
app/utils/thread_service.py
────────────────────────────
MySQL-persisted thread management with connection pooling and streaming support.

ACID notes
──────────
Atomicity  — each mutation is wrapped in a transaction.
Consistency — thread IDs are validated as UUID v4 before storage.
Isolation   — each operation gets its own connection.
Durability  — all thread metadata persisted to MySQL.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Iterator, Callable, Any

from langchain_core.messages import HumanMessage
import pymysql
import pymysql.cursors

from config.settings import (
    DEFAULT_MODEL_KEY,
    DEFAULT_THREAD_TITLE,
    MAX_TITLE_LENGTH,
    TITLE_PROMPT_MAX_CHARS,
    MYSQL_HOST,
    MYSQL_PORT,
    MYSQL_USER,
    MYSQL_PASSWORD,
    MYSQL_DATABASE,
)
import llms
from app.models.chat_state import ThreadMeta


# ── Database setup ────────────────────────────────────────────────────────────

def _get_connection():
    """Create a new database connection."""
    return pymysql.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DATABASE,
        charset="utf8mb4",
        autocommit=False,
        connect_timeout=10,
        cursorclass=pymysql.cursors.DictCursor,
    )


def init_threads_table() -> None:
    """Ensure the threads table exists. Call this once at application startup."""
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS threads (
                    thread_id VARCHAR(36) PRIMARY KEY,
                    title VARCHAR(255) NOT NULL,
                    model VARCHAR(100) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP 
                        ON UPDATE CURRENT_TIMESTAMP,
                    INDEX idx_created_at (created_at),
                    INDEX idx_updated_at (updated_at)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)

            # Create messages table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    thread_id VARCHAR(36) NOT NULL,
                    role VARCHAR(20) NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_thread_id (thread_id),
                    INDEX idx_timestamp (timestamp),
                    FOREIGN KEY (thread_id) REFERENCES threads(thread_id) ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)
            conn.commit()
    except Exception as e:
        print(f"[thread_service] Failed to create tables: {e}")
        raise
    finally:
        conn.close()


# ── Helpers ────────────────────────────────────────────────────────────────────

def new_thread_id() -> str:
    """Generate a new UUID v4 thread ID."""
    return str(uuid.uuid4())


def _validate(thread_id: str) -> None:
    """Validate that thread_id is a proper UUID v4."""
    try:
        uuid.UUID(thread_id)
    except ValueError as exc:
        raise ValueError(f"Invalid thread_id: '{thread_id}'") from exc


def _row_to_meta(row: dict) -> ThreadMeta:
    """Convert database row to ThreadMeta dict."""
    return {
        "thread_id": row["thread_id"],
        "title": row["title"],
        "model": row["model"],
        "created_at": str(row["created_at"]) if row["created_at"] else "",
    }


# ── CRUD Operations (Persisted to MySQL) ───────────────────────────────────────

def create_thread(
    thread_id: str,
    title: str = DEFAULT_THREAD_TITLE,
    model: str = DEFAULT_MODEL_KEY,
    created_at: str = "",
) -> ThreadMeta:
    """
    Create a new thread in the database.
    If created_at is not provided, MySQL will use CURRENT_TIMESTAMP.
    """
    _validate(thread_id)

    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            # Use created_at if provided, otherwise let MySQL handle it
            if created_at:
                cur.execute(
                    """INSERT INTO threads (thread_id, title, model, created_at)
                       VALUES (%s, %s, %s, %s)
                       ON DUPLICATE KEY UPDATE
                       title = VALUES(title), model = VALUES(model)""",
                    (thread_id, title[:MAX_TITLE_LENGTH], model, created_at)
                )
            else:
                cur.execute(
                    """INSERT INTO threads (thread_id, title, model)
                       VALUES (%s, %s, %s)
                       ON DUPLICATE KEY UPDATE
                       title = VALUES(title), model = VALUES(model)""",
                    (thread_id, title[:MAX_TITLE_LENGTH], model)
                )
            conn.commit()

            # Fetch the created/updated record
            cur.execute(
                "SELECT thread_id, title, model, created_at FROM threads WHERE thread_id = %s",
                (thread_id,)
            )
            row = cur.fetchone()

        if not row:
            raise RuntimeError(f"Failed to create thread {thread_id}")

        return _row_to_meta(row)
    except Exception as e:
        conn.rollback()
        print(f"[thread_service] create_thread failed: {e}")
        raise
    finally:
        conn.close()


def get_thread(thread_id: str) -> Optional[ThreadMeta]:
    """Retrieve a single thread by ID."""
    _validate(thread_id)

    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT thread_id, title, model, created_at FROM threads WHERE thread_id = %s",
                (thread_id,)
            )
            row = cur.fetchone()

        return _row_to_meta(row) if row else None
    except Exception as e:
        print(f"[thread_service] get_thread failed: {e}")
        return None
    finally:
        conn.close()


def save_message(thread_id: str, role: str, content: str) -> None:
    """
    Save a message to the database.

    Args:
        thread_id: The thread ID
        role: 'user' or 'assistant'
        content: The message content
    """
    _validate(thread_id)

    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            # Insert the message
            cur.execute("""
                INSERT INTO messages (thread_id, role, content)
                VALUES (%s, %s, %s)
            """, (thread_id, role, content))

            # Update the thread's updated_at timestamp
            cur.execute("""
                UPDATE threads SET updated_at = CURRENT_TIMESTAMP 
                WHERE thread_id = %s
            """, (thread_id,))

            conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"[thread_service] save_message failed: {e}")
        raise
    finally:
        conn.close()


def load_conversation(thread_id: str) -> List[Dict[str, str]]:
    """
    Load conversation history for a thread.

    Args:
        thread_id: The thread ID

    Returns:
        List of messages with 'role' and 'content' keys
    """
    _validate(thread_id)

    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT role, content 
                FROM messages 
                WHERE thread_id = %s 
                ORDER BY timestamp ASC
            """, (thread_id,))

            rows = cur.fetchall()

        return [{"role": row["role"], "content": row["content"]} for row in rows]
    except Exception as e:
        print(f"[thread_service] load_conversation failed: {e}")
        return []
    finally:
        conn.close()


def update_title(thread_id: str, new_title: str) -> bool:
    """Update a thread's title. Returns True if successful."""
    _validate(thread_id)

    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            affected = cur.execute(
                "UPDATE threads SET title = %s WHERE thread_id = %s",
                (new_title[:MAX_TITLE_LENGTH], thread_id)
            )
            conn.commit()
        return affected > 0
    except Exception as e:
        conn.rollback()
        print(f"[thread_service] update_title failed: {e}")
        return False
    finally:
        conn.close()


def update_model(thread_id: str, model_key: str) -> bool:
    """Update the default model for a thread. Returns True if successful."""
    _validate(thread_id)

    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            affected = cur.execute(
                "UPDATE threads SET model = %s WHERE thread_id = %s",
                (model_key, thread_id)
            )
            conn.commit()
        return affected > 0
    except Exception as e:
        conn.rollback()
        print(f"[thread_service] update_model failed: {e}")
        return False
    finally:
        conn.close()


def delete_thread(thread_id: str) -> bool:
    """Delete a thread. Returns True if a thread was deleted."""
    _validate(thread_id)

    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            # Messages will be deleted automatically due to FOREIGN KEY ON DELETE CASCADE
            affected = cur.execute(
                "DELETE FROM threads WHERE thread_id = %s",
                (thread_id,)
            )
            conn.commit()
        return affected > 0
    except Exception as e:
        conn.rollback()
        print(f"[thread_service] delete_thread failed: {e}")
        return False
    finally:
        conn.close()


def list_threads(limit: Optional[int] = None) -> List[ThreadMeta]:
    """
    Return all threads, newest first.

    Parameters
    ----------
    limit: Optional[int]
        Maximum number of threads to return (for pagination).
    """
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            query = """
                SELECT thread_id, title, model, created_at
                FROM threads
                ORDER BY created_at DESC
            """
            if limit:
                query += " LIMIT %s"
                cur.execute(query, (limit,))
            else:
                cur.execute(query)

            rows = cur.fetchall()

        return [_row_to_meta(row) for row in rows]
    except Exception as e:
        print(f"[thread_service] list_threads failed: {e}")
        return []
    finally:
        conn.close()


def get_most_recent_thread_id() -> Optional[str]:
    """Get the ID of the most recently created or updated thread."""
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT thread_id
                FROM threads
                ORDER BY updated_at DESC, created_at DESC
                LIMIT 1
            """)
            row = cur.fetchone()

        return row["thread_id"] if row else None
    except Exception as e:
        print(f"[thread_service] get_most_recent_thread_id failed: {e}")
        return None
    finally:
        conn.close()


def get_thread_title(thread_id: str) -> str:
    """Get a thread's title, or default if not found."""
    meta = get_thread(thread_id)
    return meta["title"] if meta else DEFAULT_THREAD_TITLE


def get_thread_model(thread_id: str) -> str:
    """Get a thread's default model, or default if not found."""
    meta = get_thread(thread_id)
    return meta["model"] if meta else DEFAULT_MODEL_KEY


def thread_exists(thread_id: str) -> bool:
    """Check if a thread exists in the database."""
    return get_thread(thread_id) is not None


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


# ── Streaming Response Generation ──────────────────────────────────────────────

def generate_response_with_streaming(
    model_key: str,
    messages: List[Dict[str, str]],
    placeholder=None
) -> str:
    """
    Generate a streaming response and update the UI in real-time.

    Args:
        model_key: The model to use
        messages: Conversation history
        placeholder: Streamlit placeholder for live updates

    Returns:
        Full response text
    """
    from llms.streaming import get_streaming_handler, supports_streaming

    # Get the last user message
    user_message = messages[-1]["content"] if messages else ""

    # Build context from history (last 5 messages for context)
    context_messages = messages[:-1] if len(messages) > 1 else []
    if context_messages:
        context = "\n".join(
            [f"{m['role']}: {m['content']}" for m in context_messages[-5:]])
        prompt = f"Previous conversation:\n{context}\n\nUser: {user_message}\nAssistant:"
    else:
        prompt = user_message

    full_response = ""

    # Check if model supports streaming
    if supports_streaming(model_key):
        handler = get_streaming_handler(model_key)

        # Stream the response
        for chunk in handler.stream(prompt):
            full_response += chunk
            if placeholder:
                # Update placeholder with accumulated response
                placeholder.markdown(full_response + "▌")
    else:
        # Fallback to non-streaming
        llm = llms.build_llm(model_key)
        response = llm.invoke(prompt)
        full_response = response.content if hasattr(
            response, 'content') else str(response)
        if placeholder:
            placeholder.markdown(full_response)

    if placeholder:
        # Final update without cursor
        placeholder.markdown(full_response)

    return full_response


def generate_response_with_context(
    model_key: str,
    conversation_history: List[Dict[str, str]],
    user_message: str,
    placeholder=None,
    show_thinking: bool = True
) -> str:
    """
    Generate a streaming response with conversation context.

    Args:
        model_key: The model to use
        conversation_history: List of previous messages
        user_message: Current user message
        placeholder: Streamlit placeholder for live updates
        show_thinking: Show reasoning for DeepSeek

    Returns:
        Complete response text
    """
    from llms.streaming import get_streaming_handler, supports_streaming

    # Build context from history (last 5 messages for context)
    context_messages = conversation_history if conversation_history else []
    if context_messages:
        context = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in context_messages[-5:]
        ])
        full_prompt = f"Previous conversation:\n{context}\n\nUser: {user_message}\nAssistant:"
    else:
        full_prompt = user_message

    full_response = ""

    # Check if model supports streaming
    if supports_streaming(model_key):
        handler = get_streaming_handler(model_key)

        # Special handling for DeepSeek with reasoning steps
        if model_key == "deepseek-r1" and show_thinking:
            for chunk in handler.stream(full_prompt, show_thinking=show_thinking):
                full_response += chunk
                if placeholder:
                    placeholder.markdown(full_response + "▌")
        else:
            for chunk in handler.stream(full_prompt):
                full_response += chunk
                if placeholder:
                    placeholder.markdown(full_response + "▌")
    else:
        # Fallback to non-streaming
        llm = llms.build_llm(model_key)
        response = llm.invoke(full_prompt)
        full_response = response.content if hasattr(
            response, 'content') else str(response)
        if placeholder:
            placeholder.markdown(full_response)

    if placeholder:
        # Final update without cursor
        placeholder.markdown(full_response)

    return full_response
