"""
app/core/graph.py
──────────────────
LangGraph StateGraph with MySQL checkpointing and streaming.

Architecture (refactored):
    START → router_node → direct_node   → END
                        → rag_node      → END
                        → agent_node    → END

router_node  : classifies query intent → decides execution path
             : also predicts retrieval necessity (RQ3)
direct_node  : plain LLM answer, no retrieval (uses orchestrator for model selection)
rag_node     : retrieves from ChromaDB only when router says needed (uses orchestrator)
agent_node   : DuckDuckGo search + Python calculator via ReAct (uses orchestrator)
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
from typing import Any, Iterator, Literal, Optional, Sequence, Tuple

import pymysql
import pymysql.cursors
from dotenv import load_dotenv
from pathlib import Path

# Import your orchestrator
from llms.custom.orchestrator import (
    classify_task,
    select_primary_model,
    build_specialist_prompt,
    should_synthesize,
    build_synthesis_prompt,
    build_attribution_footer,
)

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

# ── Model-specific RAG preferences ────────────────────────────────────────────
MODEL_RAG_PREFERENCES = {
    "falcon3": {"format": "plain", "strength": "high"},
    "qwen3.5-0.8b": {"format": "structured", "strength": "medium"},
    "qwen2_5_coder_7b": {"format": "structured", "strength": "medium"},
    "llama-8b-instant": {"format": "structured", "strength": "high"},
    # "gemini-2.5-flash": {"format": "plain", "strength": "high"},
    "llama3.2-3b": {"format": "structured", "strength": "medium"},
    "phi3-3.8b": {"format": "structured", "strength": "medium"},
    "granite3-dense-2b": {"format": "structured", "strength": "medium"},
    "gemma3-270m": {"format": "simple", "strength": "low"},
    "deepseek_r1": {"format": "structured", "strength": "high"},
    "mistral-7b": {"format": "structured", "strength": "high"},
}


def format_rag_context_for_model(rag_result: dict, model_key: str) -> str:
    """
    Format RAG context according to model's preferences.

    Args:
        rag_result: Output from retrieve_context()
        model_key: The model being used

    Returns:
        Formatted context string optimized for the specific model
    """
    if not rag_result.get('has_context', False):
        return ""

    pref = MODEL_RAG_PREFERENCES.get(
        model_key, {"format": "structured", "strength": "medium"})
    format_type = pref["format"]

    if format_type == "plain":
        # Simple plain text format
        return rag_result.get('formatted_text', '')

    elif format_type == "structured":
        # Structured format with clear markers
        parts = []
        parts.append("=== RELEVANT DOCUMENT EXCERPTS ===")
        parts.append("")
        for i, chunk in enumerate(rag_result.get('raw_chunks', []), 1):
            parts.append(f"[{i}] From '{chunk['source']}':")
            parts.append(chunk['content'])
            parts.append("")
        parts.append("=== END OF EXCERPTS ===")
        parts.append("")
        parts.append(
            "IMPORTANT: Answer the user's question based ONLY on the information above.")
        parts.append(
            "If the excerpts don't contain the answer, say so clearly.")
        return "\n".join(parts)

    elif format_type == "simple":
        # Very simple format for small models
        parts = []
        parts.append("Context from documents:")
        # Limit to 3 chunks for small models
        for chunk in rag_result.get('raw_chunks', [])[:3]:
            # Truncate for small models
            truncated = chunk['content'][:500] if len(
                chunk['content']) > 500 else chunk['content']
            parts.append(f"- {truncated}")
        return "\n".join(parts)

    else:
        return rag_result.get('formatted_text', '')


def format_rag_instructions(model_key: str) -> str:
    """
    Get model-specific RAG instructions.
    """
    pref = MODEL_RAG_PREFERENCES.get(
        model_key, {"format": "structured", "strength": "medium"})
    strength = pref["strength"]

    if strength == "high":
        return """
CRITICAL INSTRUCTIONS FOR USING THE PROVIDED CONTEXT:
1. The "RELEVANT DOCUMENT EXCERPTS" section contains information from uploaded documents
2. Base your answer STRICTLY on this information
3. If the excerpts don't contain the answer, say "Based on the uploaded documents, I cannot find information about this topic."
4. Do not invent or hallucinate information not present in the excerpts
5. Cite which document each piece of information comes from
"""
    elif strength == "medium":
        return """
Instructions: Use the document excerpts above to answer the user's question.
If the information isn't in the excerpts, say so honestly.
"""
    else:
        return "Use the context above to answer the question if relevant."


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


# ── Helpers ───────────────────────────────────────────────────────────────────

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

    while trimmed and not isinstance(trimmed[0], HumanMessage):
        trimmed.pop(0)

    if not trimmed:
        trimmed = [messages[-1]]

    return trimmed


TOKEN_LIMITS: dict[str, int] = {
    "llama-8b-instant":  4000,
    # "gemini-2.5-flash":  8000,
    "llama3.2-3b":       3000,
    "qwen3.5-0.8b":      3000,
    "phi3-3.8b":         3000,
    "granite3-dense-2b": 3000,
    "gemma3-270m":       2000,
    "qwen2_5_coder_7b":  4000,
    "deepseek_r1":       8000,
    "falcon3":           3000,
    "mistral-7b":        4000,
}

# Keywords that strongly signal web/real-time information need
_AGENT_KEYWORDS = (
    "latest", "current", "today", "news", "recent", "now", "live",
    "price", "weather", "stock", "score", "calculate", "compute",
    "how much", "convert", "solve",
)

# Keywords that signal document retrieval need
_RAG_KEYWORDS = (
    "document", "uploaded", "file", "pdf", "according to", "in the",
    "what does it say", "from the", "based on the", "summary", "summarize",
    "summarise", "mentioned", "context", "explain from", "what does",
    "tell me about", "describe", "what is in", "find in", "search the",
    "extract", "highlight", "key points", "main points", "outline",
)


def _get_user_message(state: ChatState) -> str | None:
    """Extract the latest HumanMessage content from state."""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            return msg.content
    return None


def _build_history_block(messages: list, model_key: str) -> str:
    """Trim and format conversation history as a prompt block."""
    max_tokens = TOKEN_LIMITS.get(model_key, 3000)
    trimmed = _trim_messages(messages, max_tokens=max_tokens)
    lines = []
    for msg in trimmed:
        if isinstance(msg, HumanMessage):
            lines.append(f"User: {msg.content}")
        elif isinstance(msg, AIMessage):
            lines.append(f"Assistant: {msg.content}")
    return "\n".join(lines)


def _call_llm_with_orchestrator(
    query: str,
    history_block: str,
    model_key: str = None,
    rag_context: str = None
) -> tuple[str, dict]:
    """
    Use orchestrator for intelligent model selection and response generation.

    Returns:
        tuple: (response_content, routing_metadata)
    """
    # Step 1: Classify task and complexity using orchestrator
    categories, complexity = classify_task(query)

    # Step 2: Select the best specialist model (or use provided model_key)
    if model_key:
        selected_model = model_key
        auto_selected = (model_key == DEFAULT_MODEL_KEY)
    else:
        selected_model = select_primary_model(categories, complexity)
        auto_selected = True

    # Step 3: Build specialist prompt
    enhanced_prompt = build_specialist_prompt(
        query, categories, complexity, selected_model
    )

    # Step 4: Add RAG context if provided
    if rag_context:
        enhanced_prompt = f"CONTEXT FROM DOCUMENTS:\n{rag_context}\n\n{enhanced_prompt}"

    # Step 5: Add conversation history
    if history_block:
        enhanced_prompt = f"CONVERSATION HISTORY:\n{history_block}\n\n{enhanced_prompt}"

    # Step 6: Invoke the selected model
    try:
        llm = llms.build_llm(selected_model)
        response = llm.invoke(enhanced_prompt)
        raw_response = response.content

        # Step 7: Optionally synthesize/refine the response
        if should_synthesize(categories, complexity):
            synthesis_prompt = build_synthesis_prompt(
                query, raw_response, categories, selected_model
            )
            synthesis_llm = llms.build_llm(
                "llama-3.1-8b-instant")
            refined = synthesis_llm.invoke(synthesis_prompt)
            final_response = refined.content
            synthesis_used = True
        else:
            final_response = raw_response
            synthesis_used = False

        # Step 8: Build attribution footer
        footer = build_attribution_footer(
            categories=categories,
            complexity=complexity,
            primary_model=selected_model,
            synthesis_model="llama-3.1-8b-instant" if synthesis_used else None,
            fallback_used=False
        )

        final_response_with_footer = final_response + footer

        routing_metadata = {
            "model_key": selected_model,
            "model_name": llms.get_display_name(selected_model),
            "auto_selected": auto_selected,
            "categories": categories,
            "complexity": complexity,
            "synthesis_used": synthesis_used,
            "reason": f"Orchestrator selected {selected_model} for {', '.join(categories)} task"
        }

        return final_response_with_footer, routing_metadata

    except Exception as e:
        print(f"[_call_llm_with_orchestrator] Error: {e}")
        fallback_llm = llms.build_llm(DEFAULT_MODEL_KEY)
        response = fallback_llm.invoke(query)
        return response.content, {
            "model_key": DEFAULT_MODEL_KEY,
            "model_name": llms.get_display_name(DEFAULT_MODEL_KEY),
            "auto_selected": False,
            "error": str(e),
            "reason": f"Fallback due to error: {str(e)[:100]}"
        }


# ── Route Classification ──────────────────────────────────────────────────────

RouteLabel = Literal["direct", "rag", "agent"]


def _classify_route(query: str, has_documents: bool) -> RouteLabel:
    """
    Retrieval necessity predictor (RQ3) + agent trigger.
    Uses orchestrator's task classification to make smarter decisions.

    Priority:
      1. rag     — documents exist AND query clearly targets them (checked FIRST)
      2. agent   — query needs real-time data or calculation
      3. direct  — everything else

    RAG is checked before agent to prevent document queries being
    incorrectly hijacked by broad agent keywords.
    """
    q_lower = query.lower()

    categories, complexity = classify_task(query)

    if has_documents:
        if "data_extraction" in categories or "summarization" in categories:
            return "rag"
        if any(kw in q_lower for kw in _RAG_KEYWORDS):
            return "rag"

    if any(kw in q_lower for kw in _AGENT_KEYWORDS):
        return "agent"

    return "direct"


# ── Node: router ──────────────────────────────────────────────────────────────

def router_node(state: ChatState) -> dict:
    """
    Classifies the query and writes 'route' into state.
    Uses orchestrator for enhanced classification.
    """
    query = _get_user_message(state) or ""
    model_key = state.get("model") or DEFAULT_MODEL_KEY

    has_documents = False
    try:
        from app.rag.retriever import get_vectorstore_count
        has_documents = get_vectorstore_count() > 0
    except Exception:
        has_documents = False

    categories, complexity = classify_task(query)
    route = _classify_route(query, has_documents)

    routing_info: RoutingInfo = {
        "model_key":     model_key,
        "model_name":    llms.get_display_name(model_key),
        "reason":        f"Task: {', '.join(categories)} (complexity: {complexity}) → routed to [{route}] path.",
        "auto_selected": True,
    }

    print(
        f"[router_node] query='{query[:60]}...' | categories={categories} | complexity={complexity} → route={route}")

    return {
        "routing_info": routing_info,
        "route":        route,
        "rag_context":  None,
        "orchestrator_info": {
            "categories": categories,
            "complexity": complexity,
        }
    }


def _route_selector(state: ChatState) -> RouteLabel:
    """Edge function: reads route from state to drive conditional branching."""
    return state.get("route", "direct")


# ── Node: direct ──────────────────────────────────────────────────────────────

def direct_node(state: ChatState) -> dict:
    """
    Plain LLM answer — no retrieval, no tools.
    Uses orchestrator for intelligent model selection and response generation.
    """
    query = _get_user_message(state)
    if not query:
        return {
            "messages":    [AIMessage(content="I didn't receive a message.")],
            "rag_context": None,
        }

    model_key = state.get("model") or DEFAULT_MODEL_KEY
    history_block = _build_history_block(state['messages'], model_key)

    response_content, routing_metadata = _call_llm_with_orchestrator(
        query=query,
        history_block=history_block,
        model_key=model_key,
        rag_context=None
    )

    routing_info = state.get("routing_info", {})
    routing_info.update(routing_metadata)

    return {
        "messages":    [AIMessage(content=response_content)],
        "rag_context": None,
        "routing_info": routing_info,
    }


# ── Node: rag ────────────────────────────────────────────────────────────────

def rag_node(state: ChatState) -> dict:
    """
    Retrieval-augmented node — only reached when router says 'rag'.
    Uses orchestrator for intelligent model selection with enhanced RAG formatting.
    """
    query = _get_user_message(state)
    if not query:
        return {
            "messages":    [AIMessage(content="I didn't receive a message.")],
            "rag_context": None,
        }

    model_key = state.get("model") or DEFAULT_MODEL_KEY

    # Retrieve context with structured result
    rag_context_result = {"has_context": False,
                          "formatted_text": "", "count": 0, "raw_chunks": []}
    try:
        from app.rag.retriever import retrieve_context
        rag_context_result = retrieve_context(query, model_key=model_key)
    except RuntimeError as rag_err:
        return {
            "messages": [AIMessage(
                content=(
                    f"⚠️ **Document retrieval failed** — I cannot access "
                    f"the uploaded documents right now.\n\n"
                    f"**Reason:** {str(rag_err)}\n\n"
                    f"Please make sure Ollama is running:\n"
                    f"```\nollama serve\nollama pull nomic-embed-text\n```"
                )
            )],
            "rag_context": None,
        }
    except Exception as e:
        print(f"[rag_node] Unexpected RAG error: {e}")
        rag_context_result = {"has_context": False,
                              "formatted_text": "", "count": 0, "raw_chunks": []}

    history_block = _build_history_block(state['messages'], model_key)

    # Get model-specific context formatting
    formatted_context = format_rag_context_for_model(
        rag_context_result, model_key)
    rag_instructions = format_rag_instructions(model_key)

    if rag_context_result.get('has_context', False):
        # Build enhanced prompt with clear RAG instructions
        enhanced_query = f"""{rag_instructions}

{formatted_context}

User Question: {query}

Now provide your answer based on the document excerpts above:"""

        # Use orchestrator with RAG context
        response_content, routing_metadata = _call_llm_with_orchestrator(
            query=enhanced_query,
            history_block=history_block,
            model_key=model_key,
            rag_context=None  # Already embedded in enhanced_query
        )
    else:
        # No relevant context found — be explicit about it
        no_context_msg = f"""I searched the uploaded documents but couldn't find information related to "{query}".

Based on the documents I have access to, I cannot answer this question. Could you please:
1. Rephrase your question, or
2. Upload more relevant documents

Is there something else I can help you with regarding the existing documents?"""

        # Still use orchestrator for natural response
        response_content, routing_metadata = _call_llm_with_orchestrator(
            query=no_context_msg,
            history_block=history_block,
            model_key=model_key,
            rag_context=None
        )

    # Update routing_info
    routing_info = state.get("routing_info", {})
    routing_info.update(routing_metadata)
    routing_info["rag_used"] = rag_context_result.get('has_context', False)
    routing_info["rag_chunks_used"] = rag_context_result.get('count', 0)

    return {
        "messages":    [AIMessage(content=response_content)],
        "rag_context": rag_context_result.get('formatted_text') if rag_context_result.get('has_context') else None,
        "routing_info": routing_info,
    }


# ── Node: agent ───────────────────────────────────────────────────────────────

def agent_node(state: ChatState) -> dict:
    """
    Minimal ReAct agent with two tools:
      1. DuckDuckGoSearchRun — free web search, no API key needed
      2. PythonCalculator    — safe arithmetic via ast.literal_eval / numexpr

    Uses orchestrator for the agent's internal LLM.
    """
    query = _get_user_message(state)
    if not query:
        return {
            "messages":    [AIMessage(content="I didn't receive a message.")],
            "rag_context": None,
        }

    model_key = state.get("model") or DEFAULT_MODEL_KEY

    categories, complexity = classify_task(query)
    recommended_model = select_primary_model(categories, complexity)

    agent_model = recommended_model if model_key == DEFAULT_MODEL_KEY else model_key

    try:
        from langchain_community.tools import DuckDuckGoSearchRun
        from langchain.tools import tool
        from langchain.agents import AgentExecutor, create_react_agent
        from langchain_core.prompts import PromptTemplate
        import ast
        import operator as _op

        search_tool = DuckDuckGoSearchRun()

        _SAFE_OPS = {
            ast.Add:  _op.add,
            ast.Sub:  _op.sub,
            ast.Mult: _op.mul,
            ast.Div:  _op.truediv,
            ast.Pow:  _op.pow,
            ast.Mod:  _op.mod,
            ast.USub: _op.neg,
        }

        def _safe_eval(node):
            if isinstance(node, ast.Constant):
                return node.n
            if isinstance(node, ast.BinOp):
                op_fn = _SAFE_OPS.get(type(node.op))
                if op_fn is None:
                    raise ValueError(f"Unsupported operator: {node.op}")
                return op_fn(_safe_eval(node.left), _safe_eval(node.right))
            if isinstance(node, ast.UnaryOp):
                op_fn = _SAFE_OPS.get(type(node.op))
                if op_fn is None:
                    raise ValueError(f"Unsupported operator: {node.op}")
                return op_fn(_safe_eval(node.operand))
            raise ValueError(f"Unsupported expression type: {type(node)}")

        @tool
        def calculator(expression: str) -> str:
            """
            Evaluates a safe arithmetic expression.
            Supports +, -, *, /, **, %.
            Example: '(3 + 5) * 2' → '16'
            """
            try:
                tree = ast.parse(expression, mode="eval")
                result = _safe_eval(tree.body)
                return str(result)
            except Exception as e:
                return f"Calculator error: {e}"

        tools = [search_tool, calculator]

        react_prompt = PromptTemplate.from_template(
            "You are a helpful assistant with access to tools.\n\n"
            "Task classification: {categories} (complexity: {complexity})\n"
            "Recommended model: {recommended_model}\n\n"
            "Tools available:\n{tools}\n\n"
            "Tool names: {tool_names}\n\n"
            "Use this format:\n"
            "Thought: <your reasoning>\n"
            "Action: <tool name>\n"
            "Action Input: <input to tool>\n"
            "Observation: <tool result>\n"
            "... (repeat Thought/Action/Observation as needed)\n"
            "Thought: I now know the final answer\n"
            "Final Answer: <your answer>\n\n"
            "Previous conversation:\n"
            f"{_build_history_block(state['messages'], agent_model)}\n\n"
            "Question: {input}\n\n"
            "{agent_scratchpad}"
        )

        llm = llms.build_llm(agent_model)
        agent = create_react_agent(llm, tools, react_prompt)
        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            max_iterations=5,
            handle_parsing_errors=True,
            verbose=False,
        )

        result = executor.invoke({
            "input": query,
            "categories": ", ".join(categories),
            "complexity": complexity,
            "recommended_model": recommended_model,
        })
        answer = result.get("output", "No answer returned by agent.")

        footer = build_attribution_footer(
            categories=categories,
            complexity=complexity,
            primary_model=agent_model,
            synthesis_model=None,
            fallback_used=False
        )

        final_answer = answer + footer

        routing_info = state.get("routing_info", {})
        routing_info.update({
            "model_key": agent_model,
            "model_name": llms.get_display_name(agent_model),
            "auto_selected": agent_model == recommended_model,
            "categories": categories,
            "complexity": complexity,
            "reason": f"Agent used {agent_model} for {', '.join(categories)} task"
        })

        return {
            "messages":    [AIMessage(content=final_answer)],
            "rag_context": None,
            "routing_info": routing_info,
        }

    except ImportError as e:
        print(f"[agent_node] Import error — falling back to direct: {e}")
        return direct_node(state)

    except Exception as exc:
        print(f"[agent_node] Agent error: {exc}")
        return direct_node(state)


# ── Graph compilation ─────────────────────────────────────────────────────────

def _compile():
    try:
        builder = StateGraph(ChatState)

        builder.add_node("router_node", router_node)
        builder.add_node("direct_node", direct_node)
        builder.add_node("rag_node",    rag_node)
        builder.add_node("agent_node",  agent_node)

        builder.add_edge(START, "router_node")

        builder.add_conditional_edges(
            "router_node",
            _route_selector,
            {
                "direct": "direct_node",
                "rag":    "rag_node",
                "agent":  "agent_node",
            },
        )

        builder.add_edge("direct_node", END)
        builder.add_edge("rag_node",    END)
        builder.add_edge("agent_node",  END)

        return builder.compile(checkpointer=_checkpointer)

    except Exception as e:
        print(f"[graph.py] Failed to compile graph: {e}")
        builder = StateGraph(ChatState)
        builder.add_node("router_node", router_node)
        builder.add_node("direct_node", direct_node)
        builder.add_node("rag_node",    rag_node)
        builder.add_node("agent_node",  agent_node)
        builder.add_edge(START, "router_node")
        builder.add_conditional_edges(
            "router_node",
            _route_selector,
            {
                "direct": "direct_node",
                "rag":    "rag_node",
                "agent":  "agent_node",
            },
        )
        builder.add_edge("direct_node", END)
        builder.add_edge("rag_node",    END)
        builder.add_edge("agent_node",  END)
        return builder.compile()


chatbot = _compile()
