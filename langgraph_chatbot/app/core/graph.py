"""
app/core/graph.py
──────────────────
LangGraph StateGraph with MySQL checkpointing and streaming.
RAG is handled inside the chat_node – it retrieves context and builds a
proper prompt with instructions to cite sources.
"""

from __future__ import annotations
from config.settings import (
    DEFAULT_MODEL_KEY,
    MYSQL_HOST,
    MYSQL_PORT,
    MYSQL_USER,
    MYSQL_PASSWORD,
    MYSQL_DATABASE,
)
from app.models.chat_state import ChatState, RoutingInfo
import llms
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)
from langgraph.graph import END, START, StateGraph
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage

import json
import threading
import time
from typing import Any, Iterator, Optional, Sequence, Tuple

import pymysql
import pymysql.cursors
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).resolve().parents[3] / ".env")


# ── Serialization ─────────────────────────────────────────────────────────────

def _msg_to_dict(msg: BaseMessage) -> dict:
    return {
        "__msg_type__": type(msg).__name__,
        "content":      msg.content,
    }


def _dict_to_msg(d: dict) -> BaseMessage:
    if d.get("__msg_type__") == "HumanMessage":
        return HumanMessage(content=d["content"])
    return AIMessage(content=d["content"])


def _serialize(obj: Any) -> bytes:
    def _convert(o: Any) -> Any:
        if isinstance(o, BaseMessage):
            return _msg_to_dict(o)
        if isinstance(o, dict):
            return {k: _convert(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_convert(i) for i in o]
        return o
    return json.dumps(_convert(obj)).encode("utf-8")


def _deserialize(data: bytes | str) -> Any:
    def _restore(o: Any) -> Any:
        if isinstance(o, dict):
            if "__msg_type__" in o:
                return _dict_to_msg(o)
            return {k: _restore(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_restore(i) for i in o]
        return o
    raw = json.loads(
        data.decode("utf-8") if isinstance(data, (bytes, bytearray)) else data
    )
    return _restore(raw)


# ── MySQL Checkpointer ────────────────────────────────────────────────────────

class MySQLSaver(BaseCheckpointSaver):
    """MySQL checkpointer with thread-local connection management."""

    def __init__(self) -> None:
        super().__init__()
        self._local = threading.local()

    def _get_conn(self):
        conn = getattr(self._local, 'conn', None)
        if conn is None:
            self._local.conn = self._create_connection()
        return self._local.conn

    def _create_connection(self):
        try:
            conn = pymysql.connect(
                host=MYSQL_HOST,
                port=MYSQL_PORT,
                user=MYSQL_USER,
                password=MYSQL_PASSWORD,
                database=MYSQL_DATABASE,
                charset="utf8mb4",
                autocommit=False,
                connect_timeout=10,
                cursorclass=pymysql.cursors.Cursor
            )
            print("[MySQLSaver] Database connected successfully")
            return conn
        except Exception as e:
            print(f"[MySQLSaver] Connection failed: {e}")
            raise

    def _ensure_connection(self):
        conn = getattr(self._local, 'conn', None)

        if conn is None:
            self._local.conn = self._create_connection()
            return self._local.conn

        try:
            conn.cursor().execute("SELECT 1")
            return conn
        except Exception as e:
            print(f"[MySQLSaver] Connection lost, reconnecting... Error: {e}")
            try:
                conn.close()
            except:
                pass
            self._local.conn = None
            self._local.conn = self._create_connection()
            return self._local.conn

    def _execute_with_retry(self, operation, *args, **kwargs):
        max_retries = 3
        retry_delay = 0.5

        for attempt in range(max_retries):
            try:
                conn = self._ensure_connection()
                result = operation(conn, *args, **kwargs)
                return result
            except Exception as e:
                print(
                    f"[MySQLSaver] Operation failed (attempt {attempt + 1}/{max_retries}): {e}")
                try:
                    old_conn = getattr(self._local, 'conn', None)
                    if old_conn:
                        old_conn.close()
                except:
                    pass
                self._local.conn = None

                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                else:
                    raise

    def get_tuple(self, config: dict) -> Optional[CheckpointTuple]:
        def _get_tuple(conn):
            thread_id = config["configurable"]["thread_id"]
            checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
            checkpoint_id = config["configurable"].get("checkpoint_id")

            with conn.cursor() as cur:
                if checkpoint_id:
                    cur.execute(
                        """SELECT checkpoint, metadata, parent_id, checkpoint_id
                           FROM checkpoints
                           WHERE thread_id=%s AND checkpoint_ns=%s
                           AND checkpoint_id=%s""",
                        (thread_id, checkpoint_ns, checkpoint_id),
                    )
                else:
                    cur.execute(
                        """SELECT checkpoint, metadata, parent_id, checkpoint_id
                           FROM checkpoints
                           WHERE thread_id=%s AND checkpoint_ns=%s
                           ORDER BY checkpoint_id DESC LIMIT 1""",
                        (thread_id, checkpoint_ns),
                    )
                row = cur.fetchone()

            if not row:
                return None

            checkpoint_data, metadata_data, parent_id, cp_id = row
            return CheckpointTuple(
                config={
                    "configurable": {
                        "thread_id":     thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": cp_id,
                    }
                },
                checkpoint=_deserialize(checkpoint_data),
                metadata=_deserialize(metadata_data),
                parent_config={
                    "configurable": {
                        "thread_id":     thread_id,
                        "checkpoint_ns": checkpoint_ns,
                        "checkpoint_id": parent_id,
                    }
                } if parent_id else None,
            )

        try:
            return self._execute_with_retry(_get_tuple)
        except Exception as e:
            print(f"[MySQLSaver.get_tuple] Error: {e}")
            return None

    def list(
        self,
        config: dict,
        *,
        filter: Optional[dict] = None,
        before: Optional[dict] = None,
        limit: Optional[int] = None,
    ) -> Iterator[CheckpointTuple]:
        def _list(conn):
            thread_id = config["configurable"]["thread_id"]
            checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
            params: list = [thread_id, checkpoint_ns]

            query = """SELECT checkpoint, metadata, parent_id, checkpoint_id
                       FROM checkpoints
                       WHERE thread_id=%s AND checkpoint_ns=%s
                       ORDER BY checkpoint_id DESC"""
            if limit:
                query += " LIMIT %s"
                params.append(limit)

            with conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()

            for checkpoint_data, metadata_data, parent_id, cp_id in rows:
                yield CheckpointTuple(
                    config={
                        "configurable": {
                            "thread_id":     thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": cp_id,
                        }
                    },
                    checkpoint=_deserialize(checkpoint_data),
                    metadata=_deserialize(metadata_data),
                    parent_config={
                        "configurable": {
                            "thread_id":     thread_id,
                            "checkpoint_ns": checkpoint_ns,
                            "checkpoint_id": parent_id,
                        }
                    } if parent_id else None,
                )

        try:
            conn = self._ensure_connection()
            yield from _list(conn)
        except Exception as e:
            print(f"[MySQLSaver.list] Error: {e}")
            return

    def put(
        self,
        config: dict,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: Any,
    ) -> dict:
        def _put(conn):
            thread_id = config["configurable"]["thread_id"]
            checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
            checkpoint_id = checkpoint["id"]
            parent_id = config["configurable"].get("checkpoint_id")

            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO checkpoints
                           (thread_id, checkpoint_ns, checkpoint_id,
                            parent_id, checkpoint, metadata)
                       VALUES (%s, %s, %s, %s, %s, %s)
                       ON DUPLICATE KEY UPDATE
                           checkpoint = VALUES(checkpoint),
                           metadata   = VALUES(metadata)""",
                    (
                        thread_id,
                        checkpoint_ns,
                        checkpoint_id,
                        parent_id,
                        _serialize(checkpoint),
                        _serialize(metadata),
                    ),
                )
            conn.commit()

            return {
                "configurable": {
                    "thread_id":     thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": checkpoint_id,
                }
            }

        return self._execute_with_retry(_put)

    def put_writes(
        self,
        config: dict,
        writes: Sequence[Tuple[str, Any]],
        task_id: str,
    ) -> None:
        def _put_writes(conn):
            thread_id = config["configurable"]["thread_id"]
            checkpoint_ns = config["configurable"].get("checkpoint_ns", "")
            checkpoint_id = config["configurable"]["checkpoint_id"]

            with conn.cursor() as cur:
                for idx, (channel, value) in enumerate(writes):
                    cur.execute(
                        """INSERT INTO checkpoint_writes
                               (thread_id, checkpoint_ns, checkpoint_id,
                                task_id, idx, channel, type, value)
                           VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                           ON DUPLICATE KEY UPDATE
                               channel = VALUES(channel),
                               type    = VALUES(type),
                               value   = VALUES(value)""",
                        (
                            thread_id,
                            checkpoint_ns,
                            checkpoint_id,
                            task_id,
                            idx,
                            channel,
                            "json",
                            _serialize(value),
                        ),
                    )
            conn.commit()

        try:
            self._execute_with_retry(_put_writes)
        except Exception as e:
            print(f"[MySQLSaver.put_writes] Error: {e}")

    def delete_thread(self, thread_id: str) -> None:
        def _delete_thread(conn):
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM checkpoints WHERE thread_id=%s", (thread_id,))
                cur.execute(
                    "DELETE FROM checkpoint_writes WHERE thread_id=%s", (thread_id,))
            conn.commit()

        try:
            self._execute_with_retry(_delete_thread)
        except Exception as e:
            print(f"[MySQLSaver.delete_thread] Error: {e}")
            raise


_checkpointer = MySQLSaver()


def _trim_messages(messages: list, max_tokens: int = 4000) -> list:
    """Trim message history to stay within token limits."""
    if not messages:
        return messages

    def estimate_tokens(msg) -> int:
        return len(msg.content) // 4 + 10

    trimmed = []
    total_tokens = 0

    for msg in reversed(messages):
        msg_tokens = estimate_tokens(msg)
        if total_tokens + msg_tokens > max_tokens:
            break
        trimmed.insert(0, msg)
        total_tokens += msg_tokens

    # Ensure the first message is a HumanMessage
    while trimmed and not isinstance(trimmed[0], HumanMessage):
        trimmed.pop(0)

    if not trimmed:
        trimmed = [messages[-1]]

    return trimmed


def chat_node(state: ChatState) -> dict:
    try:
        messages = state["messages"]
        model_key = state.get("model") or DEFAULT_MODEL_KEY

        # Extract the original user message (the latest HumanMessage)
        original_user_message: str | None = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                original_user_message = msg.content
                break

        if not original_user_message:
            return {
                "messages":     [AIMessage(content="I didn't receive a message.")],
                "routing_info": None,
                "rag_context":  None,
            }

        routing_info: RoutingInfo = {
            "model_key":     model_key,
            "model_name":    llms.get_display_name(model_key),
            "reason":        "User-selected model.",
            "auto_selected": False,
        }

        # ── RAG: retrieve relevant context using the original user message ──
        rag_context = ""
        try:
            from app.rag.retriever import retrieve_context
            rag_context = retrieve_context(original_user_message)
        except RuntimeError as rag_err:
            error_msg = str(rag_err)
            print(f"[graph.chat_node] RAG retrieval failed: {error_msg}")
            return {
                "messages": [AIMessage(
                    content=(
                        f"⚠️ **Document retrieval failed** — I cannot access "
                        f"the uploaded documents right now.\n\n"
                        f"**Reason:** {error_msg}\n\n"
                        f"Please make sure Ollama is running:\n"
                        f"```\nollama serve\nollama pull nomic-embed-text\n```"
                    )
                )],
                "routing_info": routing_info,
                "rag_context":  None,
            }
        except Exception as rag_err:
            print(
                f"[graph.chat_node] RAG retrieval unexpected error: {rag_err}")
            rag_context = ""

        # ── Build the final prompt with clear instructions ────────────────
        prompt_lines = []

        if rag_context:
            prompt_lines.append(
                "You are a helpful assistant. Use the following context from "
                "uploaded documents to answer the user's question.\n"
                "**IMPORTANT INSTRUCTIONS:**\n"
                "1. If the answer is found in the context, cite the source document "
                "   (e.g., 'According to [filename]...').\n"
                "2. If the context does not contain the answer, say:\n"
                "   'Based on the uploaded documents, I couldn't find information about this. "
                "Here's my general knowledge:'\n"
                "3. Be concise and helpful.\n"
            )
            prompt_lines.append(f"CONTEXT:\n{rag_context}\n")
        else:
            prompt_lines.append(
                "You are a helpful assistant. No documents have been uploaded or no relevant "
                "context was found. Answer using your own knowledge."
            )

        prompt_lines.append("\nCONVERSATION HISTORY:")

        # Trim conversation history to fit model token limits
        token_limits = {
            "llama-8b-instant":  4000,
            "gemini-2.5-flash":  8000,
            "llama3.2-3b":       3000,
            "qwen3.5-0.8b":      3000,
            "phi3-3.8b":         3000,
            "granite3-dense-2b": 3000,
            "gemma3-270m":       2000,
        }
        max_tokens = token_limits.get(model_key, 3000)
        trimmed = _trim_messages(messages, max_tokens=max_tokens)

        for msg in trimmed:
            if isinstance(msg, HumanMessage):
                prompt_lines.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                prompt_lines.append(f"Assistant: {msg.content}")

        prompt_lines.append("Assistant:")
        prompt_text = "\n".join(prompt_lines)

        # Invoke the LLM
        try:
            llm = llms.build_llm(model_key)
            response = llm.invoke(prompt_text)
            return {
                "messages":     [AIMessage(content=response.content)],
                "routing_info": routing_info,
                "rag_context":  None,   # cleared after use
            }
        except Exception as exc:
            print(f"[graph.chat_node] LLM Error: {exc}")
            return {
                "messages":     [AIMessage(content=f"⚠️ Error: {str(exc)}")],
                "routing_info": routing_info,
                "rag_context":  None,
            }

    except Exception as e:
        print(f"[graph.chat_node] Unexpected error: {e}")
        return {
            "messages":     [AIMessage(content=f"⚠️ Unexpected error: {str(e)}")],
            "routing_info": None,
            "rag_context":  None,
        }


def _compile():
    try:
        builder = StateGraph(ChatState)
        builder.add_node("chat_node", chat_node)
        builder.add_edge(START, "chat_node")
        builder.add_edge("chat_node", END)
        return builder.compile(checkpointer=_checkpointer)
    except Exception as e:
        print(f"[graph.py] Failed to compile graph: {e}")
        # Fallback – compile without checkpointer
        builder = StateGraph(ChatState)
        builder.add_node("chat_node", chat_node)
        builder.add_edge(START, "chat_node")
        builder.add_edge("chat_node", END)
        return builder.compile()


chatbot = _compile()
