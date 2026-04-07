
from __future__ import annotations
import httpx
from langchain_ollama import ChatOllama
from .config import MODEL_ID, TEMPERATURE, MAX_TOKENS


def build() -> ChatOllama:
    """
    DeepSeek-R1 — chain-of-thought reasoning model.
    Runs locally via Ollama. No API key required.
    Note: responses are slower due to thinking steps — this is expected.
    """
    try:
        httpx.get("http://localhost:11434", timeout=3.0)
    except httpx.ConnectError:
        raise EnvironmentError(
            f"Cannot connect to Ollama at http://localhost:11434. "
            f"Please run 'ollama serve' and then 'ollama pull {MODEL_ID}'."
        )

    return ChatOllama(
        model=MODEL_ID,
        temperature=TEMPERATURE,
        num_predict=MAX_TOKENS,
    )
