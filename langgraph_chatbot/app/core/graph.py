"""
app/core/graph.py
──────────────────
LangGraph StateGraph with MySQL checkpointing and streaming.
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
from typing import Any, Iterator, Optional, Sequence, Tuple
import time

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
    """MySQL checkpointer with connection management and retry logic"""

    def __init__(self) -> None:
        super().__init__()
        self._conn = None
        self._connect()
        # Allow the connection handshake to fully complete
        # before LangGraph immediately calls get_tuple on startup
        time.sleep(0.3)

    def _connect(self) -> None:
        """Create a new database connection"""
        try:
            if self._conn:
                try:
                    self._conn.close()
                except:
                    pass
            self._conn = pymysql.connect(
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
        except Exception as e:
            print(f"[MySQLSaver] Connection failed: {e}")
            self._conn = None
            raise

    def _ensure_connection(self) -> None:
        """Ensure connection is alive, reconnect if needed"""
        # Guard: if _conn is None, just reconnect directly
        if self._conn is None:
            self._connect()
            return

        try:
            # Use a lightweight query instead of ping() — ping has known
            # issues with PyMySQL causing 'Packet sequence number wrong'
            # and 'NoneType has no attribute settimeout' on fresh connections
            self._conn.cursor().execute("SELECT 1")
        except Exception as e:
            print(f"[MySQLSaver] Connection lost, reconnecting... Error: {e}")
            self._conn = None
            try:
                self._connect()
            except Exception as connect_err:
                print(f"[MySQLSaver] Reconnect failed: {connect_err}")
                self._conn = None
                raise

    def _execute_with_retry(self, operation, *args, **kwargs):
        """Execute a database operation with retry logic"""
        max_retries = 3
        retry_delay = 0.5

        for attempt in range(max_retries):
            try:
                # ── KEY FIX: ensure connection BEFORE calling the operation,
                #    then pass self._conn explicitly so the operation always
                #    uses the live connection object, not a stale closure.
                self._ensure_connection()
                if self._conn is None:
                    raise RuntimeError(
                        "Connection unavailable after reconnect attempt")
                result = operation(self._conn, *args, **kwargs)
                return result
            except Exception as e:
                print(
                    f"[MySQLSaver] Operation failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    self._conn = None   # force fresh reconnect on next attempt
                    continue
                else:
                    raise

    # ── All inner functions now accept `conn` as their first argument ─────────

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
            self._ensure_connection()
            if self._conn is None:
                return
            yield from _list(self._conn)
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

# ── Connection factory ────────────────────────────────────────────────────────


_checkpointer = MySQLSaver()


# ── Chat node ─────────────────────────────────────────────────────────────────

def chat_node(state: ChatState) -> dict:
    try:
        messages = state["messages"]
        model_key = state.get("model") or DEFAULT_MODEL_KEY

        # Extract latest user message
        last_user_message: str | None = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                last_user_message = msg.content
                break

        if not last_user_message:
            return {
                "messages":     [AIMessage(content="I didn't receive a message.")],
                "routing_info": None,
            }

        routing_info: RoutingInfo = {
            "model_key":     model_key,
            "model_name":    llms.get_display_name(model_key),
            "reason":        "User-selected model.",
            "auto_selected": False,
        }

        # Build plain-text prompt from full history
        lines = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                lines.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                lines.append(f"Assistant: {msg.content}")
        lines.append("Assistant:")
        prompt_text = "\n".join(lines)

        try:
            llm = llms.build_llm(model_key)
            response = llm.invoke(prompt_text)
            return {
                "messages":     [AIMessage(content=response.content)],
                "routing_info": routing_info,
            }
        except Exception as exc:
            print(f"[graph.chat_node] LLM Error: {exc}")
            return {
                "messages":     [AIMessage(content=f"⚠️ Error: {str(exc)}")],
                "routing_info": routing_info,
            }
    except Exception as e:
        print(f"[graph.chat_node] Unexpected error: {e}")
        return {
            "messages": [AIMessage(content=f"⚠️ Unexpected error: {str(e)}")],
            "routing_info": None,
        }


# ── Compiled chatbot singleton ────────────────────────────────────────────────

def _compile():
    try:
        builder = StateGraph(ChatState)
        builder.add_node("chat_node", chat_node)
        builder.add_edge(START, "chat_node")
        builder.add_edge("chat_node", END)

        return builder.compile(checkpointer=_checkpointer)
    except Exception as e:
        print(f"[graph.py] Failed to compile graph: {e}")
        # Return a minimal graph without checkpointer
        builder = StateGraph(ChatState)
        builder.add_node("chat_node", chat_node)
        builder.add_edge(START, "chat_node")
        builder.add_edge("chat_node", END)
        return builder.compile()


chatbot = _compile()
