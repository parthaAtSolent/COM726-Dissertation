"""
llms/streaming.py
─────────────────
Streaming support for LLM models.
Handles both sync and async streaming with model-specific optimizations.
"""

from __future__ import annotations

from typing import Iterator, AsyncIterator, Optional, Callable, Union
from langchain_core.language_models.chat_models import BaseChatModel
import llms


class StreamingHandler:
    """Generic streaming handler for any registered LLM."""

    def __init__(self, model_key: str):
        """
        Initialize streaming handler for a specific model.

        Args:
            model_key: The model identifier (must be in REGISTRY)
        """
        self.model_key = model_key
        self.model = llms.build_llm(model_key)
        self._enable_streaming()

    def _enable_streaming(self):
        """Enable streaming on the model if possible."""
        if hasattr(self.model, 'streaming'):
            self.model.streaming = True
        elif hasattr(self.model, 'config') and hasattr(self.model.config, 'streaming'):
            self.model.config.streaming = True

    def stream(self, prompt: str, callback: Optional[Callable[[str], None]] = None) -> Iterator[str]:
        """
        Stream response synchronously.

        Args:
            prompt: User input prompt
            callback: Optional callback function called for each chunk

        Yields:
            Text chunks as they're generated
        """
        for chunk in self.model.stream(prompt):
            content = chunk.content if hasattr(
                chunk, 'content') else str(chunk)
            if callback:
                callback(content)
            yield content

    async def astream(self, prompt: str, callback: Optional[Callable[[str], None]] = None) -> AsyncIterator[str]:
        """
        Stream response asynchronously.

        Args:
            prompt: User input prompt
            callback: Optional callback function called for each chunk

        Yields:
            Text chunks as they're generated
        """
        async for chunk in self.model.astream(prompt):
            content = chunk.content if hasattr(
                chunk, 'content') else str(chunk)
            if callback:
                callback(content)
            yield content


class DeepSeekStreamingHandler(StreamingHandler):
    """Specialized streaming handler for DeepSeek-R1 with reasoning detection."""

    def __init__(self):
        """Initialize DeepSeek-specific streaming handler."""
        super().__init__("deepseek-r1")
        self.reasoning_buffer = ""
        self.response_buffer = ""

    def stream(self, prompt: str, show_thinking: bool = True, callback: Optional[Callable[[str], None]] = None) -> Iterator[str]:
        """
        Stream DeepSeek response with reasoning step detection.

        Args:
            prompt: User input prompt
            show_thinking: Whether to show reasoning steps
            callback: Optional callback for each chunk

        Yields:
            Text chunks with reasoning steps highlighted
        """
        for chunk in super().stream(prompt):
            # Detect reasoning patterns (customize based on DeepSeek's output format)
            if show_thinking and ('reasoning' in chunk.lower() or 'thinking' in chunk.lower()):
                formatted = f"\n🧠 Reasoning: {chunk}\n"
                if callback:
                    callback(formatted)
                yield formatted
            else:
                if callback:
                    callback(chunk)
                yield chunk

    async def astream(self, prompt: str, show_thinking: bool = True, callback: Optional[Callable[[str], None]] = None) -> AsyncIterator[str]:
        """
        Async stream with reasoning detection.
        """
        async for chunk in super().astream(prompt):
            if show_thinking and ('reasoning' in chunk.lower() or 'thinking' in chunk.lower()):
                formatted = f"\n🧠 Reasoning: {chunk}\n"
                if callback:
                    callback(formatted)
                yield formatted
            else:
                if callback:
                    callback(chunk)
                yield chunk


def get_streaming_handler(model_key: str):
    """
    Factory function to get the appropriate streaming handler for a model.

    Args:
        model_key: Model identifier

    Returns:
        Streaming handler instance
    """
    # Return specialized handler for DeepSeek
    if model_key == "deepseek-r1":
        return DeepSeekStreamingHandler()

    # Generic handler for all other models
    return StreamingHandler(model_key)


def stream_response(model_key: str, prompt: str, callback: Optional[Callable[[str], None]] = None) -> Iterator[str]:
    """
    Convenience function to stream response from any model.

    Args:
        model_key: Model identifier
        prompt: Input prompt
        callback: Optional callback for each chunk

    Returns:
        Iterator of text chunks
    """
    handler = get_streaming_handler(model_key)
    yield from handler.stream(prompt, callback)


async def astream_response(model_key: str, prompt: str, callback: Optional[Callable[[str], None]] = None) -> AsyncIterator[str]:
    """
    Async convenience function to stream response from any model.

    Args:
        model_key: Model identifier
        prompt: Input prompt
        callback: Optional callback for each chunk

    Returns:
        Async iterator of text chunks
    """
    handler = get_streaming_handler(model_key)
    async for chunk in handler.astream(prompt, callback):
        yield chunk


def supports_streaming(model_key: str) -> bool:
    """
    Check if a model supports streaming.

    Args:
        model_key: Model identifier

    Returns:
        True if streaming is supported, False otherwise
    """
    try:
        model = llms.build_llm(model_key)
        return hasattr(model, 'stream') and hasattr(model, 'astream')
    except:
        return False
