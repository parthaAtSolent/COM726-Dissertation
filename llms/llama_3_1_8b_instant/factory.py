"""
llms/llama_3_1_8b_instant/factory.py
──────────────────────────────────────
Builds and returns a configured ChatGroq instance for Llama 3.1 8B Instant.

Each LLM folder owns its own factory so:
  - The import path makes the model provenance explicit.
  - Adding/removing a model never touches shared service code.
  - Unit-testing a model in isolation is trivial.
"""

from __future__ import annotations

import os
import time
from typing import Iterator, AsyncIterator, Optional, Callable
from contextlib import asynccontextmanager

from langchain_groq import ChatGroq
from langchain_core.callbacks import CallbackManager

from .config import (
    API_KEY_ENV,
    MAX_TOKENS,
    MODEL_ID,
    TEMPERATURE,
    WEBSITE,
)


class LlamaStreamingHandler:
    """Handles streaming callbacks with custom processing"""

    def __init__(self, on_token: Optional[Callable[[str], None]] = None):
        self.on_token = on_token or (lambda x: None)
        self.full_response = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Process each token as it arrives"""
        self.full_response += token
        self.on_token(token)


def build(streaming: bool = False, on_token: Optional[Callable[[str], None]] = None) -> ChatGroq:
    """
    Build Llama 3.1 8B Instant model with optional streaming support.

    Args:
        streaming: Enable streaming mode
        on_token: Callback function for each token (only used if streaming=True)

    Raises
    ------
    EnvironmentError
        If ``GROQ_API_KEY`` is not set in the environment / .env file.
    """
    api_key = os.getenv(API_KEY_ENV, "")
    if not api_key:
        raise EnvironmentError(
            f"'{API_KEY_ENV}' is not set in your .env file.  "
            f"Get a free key at https://{WEBSITE}"
        )

    config = {
        "model": MODEL_ID,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "api_key": api_key,
    }

    if streaming:
        config["streaming"] = True
        if on_token:
            handler = LlamaStreamingHandler(on_token)
            config["callbacks"] = [handler]

    return ChatGroq(**config)


class Llama31_8BInstant:
    """
    Advanced streaming wrapper for Llama 3.1 8B Instant.
    Provides both sync and async streaming.
    """

    def __init__(self):
        self.model = build(streaming=False)  # Base model
        self.last_response = ""

    def stream(self, prompt: str) -> Iterator[str]:
        """
        Stream response with intelligent chunking.

        Args:
            prompt: User input
        """
        full_response = ""
        buffer = ""

        for chunk in self.model.stream(prompt):
            if hasattr(chunk, 'content'):
                content = chunk.content
            else:
                content = str(chunk)

            # Smart chunk splitting for better streaming feel
            buffer += content
            if len(buffer) >= 5 or any(p in buffer for p in ['.', '!', '?', '\n']):
                yield buffer
                full_response += buffer
                buffer = ""

        # Yield remaining buffer
        if buffer:
            yield buffer
            full_response += buffer

        # Track complete response
        self.last_response = full_response

    async def astream(self, prompt: str) -> AsyncIterator[str]:
        """
        Async stream with better performance for web applications.
        """
        buffer = ""

        async for chunk in self.model.astream(prompt):
            content = chunk.content if hasattr(
                chunk, 'content') else str(chunk)

            buffer += content
            if len(buffer) >= 5 or any(p in buffer for p in ['.', '!', '?', '\n']):
                yield buffer
                buffer = ""

        if buffer:
            yield buffer

    def stream_with_progress(self, prompt: str) -> Iterator[dict]:
        """
        Stream with progress metadata (token count, timing, etc.)
        """
        start_time = time.time()
        token_count = 0

        for chunk in self.model.stream(prompt):
            content = chunk.content if hasattr(
                chunk, 'content') else str(chunk)
            token_count += 1

            yield {
                "token": content,
                "token_index": token_count,
                "elapsed_ms": (time.time() - start_time) * 1000,
                "is_complete": False
            }

        # Final completion message
        yield {
            "token": "",
            "token_index": token_count,
            "elapsed_ms": (time.time() - start_time) * 1000,
            "is_complete": True,
            "total_tokens": token_count
        }

    @asynccontextmanager
    async def streaming_context(self, prompt: str):
        """
        Context manager for controlled streaming with cleanup.
        """
        try:
            async for chunk in self.astream(prompt):
                yield chunk
        except Exception as e:
            yield f"\n❌ Error: {str(e)}\n"
        finally:
            yield "\n✅ Streaming complete\n"


# Singleton instance for reuse
_llama_instance = None


def get_instance() -> Llama31_8BInstant:
    """Get or create singleton Llama31_8BInstant instance."""
    global _llama_instance
    if _llama_instance is None:
        _llama_instance = Llama31_8BInstant()
    return _llama_instance
