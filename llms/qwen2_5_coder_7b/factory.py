"""
Qwen2.5-Coder Factory
Builds the LangChain chat model for Qwen2.5-Coder.
"""

from __future__ import annotations
import httpx
import time
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage
from langchain_core.callbacks import CallbackManager
from typing import Iterator, AsyncIterator, Optional, Callable
from contextlib import asynccontextmanager
from .config import MODEL_ID, TEMPERATURE, MAX_TOKENS, SYSTEM_PROMPT


class QwenStreamingHandler:
    """Handles streaming callbacks with custom processing"""

    def __init__(self, on_token: Optional[Callable[[str], None]] = None):
        self.on_token = on_token or (lambda x: None)
        self.full_response = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Process each token as it arrives"""
        self.full_response += token
        self.on_token(token)


def build(streaming: bool = False, on_token: Optional[Callable[[str], None]] = None) -> ChatOllama:
    """
    Build Qwen2.5-Coder 7B model with optional streaming support.

    Args:
        streaming: Enable streaming mode
        on_token: Callback function for each token (only used if streaming=True)
    """
    # Check if Ollama is running
    try:
        httpx.get("http://localhost:11434", timeout=3.0)
    except httpx.ConnectError:
        raise EnvironmentError(
            f"Cannot connect to Ollama at http://localhost:11434.\n"
            f"Please run 'ollama serve' in a separate terminal.\n"
            f"Then pull the model: ollama pull {MODEL_ID}"
        )

    # Check if model exists (optional but helpful)
    try:
        response = httpx.post(
            "http://localhost:11434/api/show",
            json={"name": MODEL_ID},
            timeout=5.0
        )
        if response.status_code != 200:
            raise EnvironmentError(
                f"Model {MODEL_ID} not found.\n"
                f"Please pull it first: ollama pull {MODEL_ID}\n"
                f"This will download ~4.4GB for the 7B model."
            )
    except:
        # If check fails, still try to create the model
        pass

    config = {
        "model": MODEL_ID,
        "temperature": TEMPERATURE,
        "num_predict": MAX_TOKENS,
        "stop": ["User:", "Human:"],
    }

    if streaming:
        config["streaming"] = True
        if on_token:
            handler = QwenStreamingHandler(on_token)
            config["callbacks"] = [handler]

    return ChatOllama(**config)


def build_with_system_prompt(streaming: bool = False, on_token: Optional[Callable[[str], None]] = None) -> ChatOllama:
    """
    Build Qwen2.5-Coder with a system prompt (for chat applications).
    """
    model = build(streaming=streaming, on_token=on_token)

    # Wrap to include system prompt on every invocation
    from langchain_core.messages import SystemMessage, HumanMessage

    class SystemPromptWrapper:
        def __init__(self, model, system_prompt):
            self.model = model
            self.system_prompt = system_prompt

        def invoke(self, messages):
            # Add system prompt at the beginning if not already present
            if not messages or not isinstance(messages[0], SystemMessage):
                messages = [SystemMessage(
                    content=self.system_prompt)] + messages
            return self.model.invoke(messages)

        def stream(self, messages):
            # Add system prompt at the beginning if not already present
            if not messages or not isinstance(messages[0], SystemMessage):
                messages = [SystemMessage(
                    content=self.system_prompt)] + messages
            return self.model.stream(messages)

        def astream(self, messages):
            # Add system prompt at the beginning if not already present
            if not messages or not isinstance(messages[0], SystemMessage):
                messages = [SystemMessage(
                    content=self.system_prompt)] + messages
            return self.model.astream(messages)

        def __getattr__(self, name):
            return getattr(self.model, name)

    return SystemPromptWrapper(model, SYSTEM_PROMPT)


class Qwen25Coder7B:
    """
    Advanced streaming wrapper for Qwen2.5 Coder 7B.
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
_qwen_instance = None


def get_instance() -> Qwen25Coder7B:
    """Get or create singleton Qwen25Coder7B instance."""
    global _qwen_instance
    if _qwen_instance is None:
        _qwen_instance = Qwen25Coder7B()
    return _qwen_instance
