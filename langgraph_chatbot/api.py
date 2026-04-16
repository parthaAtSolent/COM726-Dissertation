"""
langgraph_chatbot/api.py
─────────────────────────
FastAPI entry point for the Android mobile application.
Wraps the existing LangGraph chatbot logic into REST endpoints.

Run with:
    cd langgraph_chatbot
    PYTHONPATH=. uvicorn api:app --reload --port 8000
"""

from __future__ import annotations
from langchain_core.messages import HumanMessage, AIMessage
from app.rag import list_ingested_files
from app.rag.ingestion import ingest_documents
from app.utils.thread_service import (
    init_threads_table,
    create_thread,
    list_threads,           # Changed from get_all_threads
    delete_thread,
    get_thread,
    save_message,
    load_conversation,
    update_title,
)
from app.core.graph import chatbot
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File, HTTPException

import sys
import uuid
from pathlib import Path

# ── Path setup (same pattern as main.py) ──────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
if str(_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_ROOT.parent))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="LangGraph Chatbot API",
    description="REST API for the Android mobile client — COM726 Dissertation",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── DB init on startup ────────────────────────────────────────────────────────


@app.on_event("startup")
def startup():
    init_threads_table()
    print("[API] Database initialised.")


# ── Pydantic models ───────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    thread_id: str
    message: str
    model: str = "llama-8b-instant"


class ChatResponse(BaseModel):
    response: str
    thread_id: str


class NewThreadResponse(BaseModel):
    thread_id: str
    title: str


class ThreadItem(BaseModel):
    thread_id: str
    title: str
    created_at: str = ""
    model: str = ""


class ThreadsResponse(BaseModel):
    threads: list[ThreadItem]


class MessageItem(BaseModel):
    role: str
    content: str


class ConversationResponse(BaseModel):
    thread_id: str
    messages: list[MessageItem]


# ── Chat ──────────────────────────────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Send a message and receive the assistant's reply.
    The thread_id keeps conversation history via MySQL checkpointing.
    """
    try:
        # Save user message to database
        save_message(req.thread_id, "user", req.message)

        # Invoke the chatbot
        result = chatbot.invoke(
            {
                "messages": [HumanMessage(content=req.message)],
                "model": req.model,
            },
            config={"configurable": {"thread_id": req.thread_id}},
        )

        messages = result.get("messages", [])
        last = messages[-1] if messages else None
        reply = last.content if isinstance(
            last, AIMessage) else "No response received."

        # Save assistant response to database
        save_message(req.thread_id, "assistant", reply)

        # If this is the first message in the thread, generate a title
        conv_history = load_conversation(req.thread_id)
        if len(conv_history) == 2:  # Just user message and assistant response
            from app.utils.thread_service import generate_title
            title = generate_title(req.message, req.model)
            update_title(req.thread_id, title)

        return ChatResponse(response=reply, thread_id=req.thread_id)

    except Exception as e:
        print(f"[API /chat] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Threads ───────────────────────────────────────────────────────────────────
@app.get("/threads", response_model=ThreadsResponse)
def get_threads(limit: int = 50):
    """Return all conversation threads."""
    try:
        # returns list of ThreadMeta dicts
        threads = list_threads(limit=limit)
        return ThreadsResponse(threads=[
            ThreadItem(
                thread_id=t["thread_id"],
                title=t.get("title", "New Chat"),
                created_at=t.get("created_at", ""),
                model=t.get("model", "")
            )
            for t in threads
        ])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/threads/{thread_id}")
def get_thread_details(thread_id: str):
    """Get details of a specific thread."""
    try:
        thread = get_thread(thread_id)
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found")

        messages = load_conversation(thread_id)

        return {
            "thread_id": thread["thread_id"],
            "title": thread["title"],
            "model": thread["model"],
            "created_at": thread["created_at"],
            "messages": messages
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/threads/{thread_id}/messages", response_model=ConversationResponse)
def get_thread_messages(thread_id: str):
    """Get conversation history for a specific thread."""
    try:
        # Verify thread exists
        thread = get_thread(thread_id)
        if not thread:
            raise HTTPException(status_code=404, detail="Thread not found")

        messages = load_conversation(thread_id)

        return ConversationResponse(
            thread_id=thread_id,
            messages=[MessageItem(role=m["role"], content=m["content"])
                      for m in messages]
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/threads", response_model=NewThreadResponse)
def new_thread(title: str = "New Chat", model: str = "llama-8b-instant"):
    """Create a new conversation thread and return its ID."""
    try:
        tid = str(uuid.uuid4())
        create_thread(tid, title, model)
        return NewThreadResponse(thread_id=tid, title=title)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/threads/{thread_id}/title")
def rename_thread(thread_id: str, title: str):
    """Rename a conversation thread."""
    try:
        success = update_title(thread_id, title)
        if not success:
            raise HTTPException(status_code=404, detail="Thread not found")
        return {"status": "updated", "thread_id": thread_id, "title": title}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/threads/{thread_id}")
def remove_thread(thread_id: str):
    """Delete a conversation thread and its checkpoint history."""
    try:
        success = delete_thread(thread_id)
        if not success:
            raise HTTPException(status_code=404, detail="Thread not found")
        return {"status": "deleted", "thread_id": thread_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── RAG / File upload ─────────────────────────────────────────────────────────
@app.post("/upload")
def upload_file(file: UploadFile = File(...)):
    """
    Upload a PDF or .txt file to be ingested into the RAG vectorstore.
    The file is chunked, embedded via Ollama, and stored in ChromaDB.
    """
    allowed_extensions = {".pdf", ".txt"}
    suffix = Path(file.filename).suffix.lower()

    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {allowed_extensions}",
        )

    tmp_path = f"/tmp/{file.filename}"
    try:
        with open(tmp_path, "wb") as f:
            import shutil
            shutil.copyfileobj(file.file, f)

        chunks_stored = ingest_documents([tmp_path])

        # Clean up temp file
        import os
        os.remove(tmp_path)

        return {
            "status": "ingested",
            "filename": file.filename,
            "chunks": chunks_stored,
        }
    except RuntimeError as e:
        # ingest_documents raises RuntimeError when Ollama is not running
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/files")
def get_files():
    """Return the list of files currently ingested into the RAG vectorstore."""
    try:
        return {"files": list_ingested_files()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    """Simple liveness check — useful for Railway and Android to ping."""
    return {"status": "ok", "service": "LangGraph Chatbot API"}


@app.get("/")
def root():
    """Root endpoint with API information."""
    return {
        "name": "LangGraph Chatbot API",
        "version": "1.0.0",
        "endpoints": {
            "chat": "/chat (POST)",
            "threads": "/threads (GET, POST)",
            "thread": "/threads/{thread_id} (GET, DELETE)",
            "thread_messages": "/threads/{thread_id}/messages (GET)",
            "upload": "/upload (POST)",
            "files": "/files (GET)",
            "health": "/health (GET)"
        }
    }
