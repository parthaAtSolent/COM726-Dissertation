from __future__ import annotations
import os
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.callbacks import CallbackManager
from typing import Iterator, AsyncIterator, Optional, Callable
from contextlib import asynccontextmanager
from .config import API_KEY_ENV, MAX_TOKENS, MODEL_ID, TEMPERATURE, WEBSITE


class GeminiStreamingHandler:
    """Handles streaming callbacks with custom processing"""

    def __init__(self, on_token: Optional[Callable[[str], None]] = None):
        self.on_token = on_token or (lambda x: None)
        self.full_response = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Process each token as it arrives"""
        self.full_response += token
        self.on_token(token)


def build(streaming: bool = False, on_token: Optional[Callable[[str], None]] = None) -> ChatGoogleGenerativeAI:
    """
    Build Gemini 2.5 Flash model with optional streaming support.

    Args:
        streaming: Enable streaming mode
        on_token: Callback function for each token (only used if streaming=True)
    """
    api_key = os.getenv(API_KEY_ENV, "")
    if not api_key:
        raise EnvironmentError(
            f"'{API_KEY_ENV}' is not set in your .env file. "
            f"Get a free key at https://{WEBSITE}"
        )

    try:
        config = {
            "model": MODEL_ID,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "google_api_key": api_key,
        }

        if streaming:
            config["streaming"] = True
            if on_token:
                handler = GeminiStreamingHandler(on_token)
                config["callbacks"] = [handler]

        llm = ChatGoogleGenerativeAI(**config)

        # Lightweight connectivity check — invoke with an empty prompt
        if not streaming:
            llm.invoke("hi")

        return llm

    except Exception as e:
        error_msg = str(e).lower()

        if "api key" in error_msg or "invalid" in error_msg or "unauthorized" in error_msg:
            raise EnvironmentError(
                f"Invalid or expired '{API_KEY_ENV}'. "
                f"Check your key at https://{WEBSITE}"
            ) from e

        if "quota" in error_msg or "rate" in error_msg or "429" in error_msg:
            raise EnvironmentError(
                f"Google API quota exceeded for model '{MODEL_ID}'. "
                f"Check your usage at https://{WEBSITE}"
            ) from e

        if "connection" in error_msg or "network" in error_msg or "timeout" in error_msg:
            raise EnvironmentError(
                f"Cannot reach Google API. Check your internet connection."
            ) from e

        # Re-raise anything else with context
        raise EnvironmentError(
            f"Failed to initialise '{MODEL_ID}': {e}"
        ) from e


class Gemini25Flash:
    """
    Advanced streaming wrapper for Gemini 2.5 Flash.
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
_gemini_instance = None


def get_instance() -> Gemini25Flash:
    """Get or create singleton Gemini25Flash instance."""
    global _gemini_instance
    if _gemini_instance is None:
        _gemini_instance = Gemini25Flash()
    return _gemini_instance
