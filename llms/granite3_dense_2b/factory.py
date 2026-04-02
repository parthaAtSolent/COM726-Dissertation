from __future__ import annotations
import httpx
from langchain_ollama import ChatOllama
from .config import MODEL_ID, TEMPERATURE, MAX_TOKENS


def build() -> ChatOllama:
    """
    Instantiate a local Ollama model. No API key required.
    Ollama must be running locally (http://localhost:11434).
    """
    # Check Ollama is reachable before returning the client
    try:
        response = httpx.get("http://localhost:11434", timeout=3.0)
    except httpx.ConnectError:
        raise EnvironmentError(
            f"Cannot connect to Ollama at http://localhost:11434. "
            f"Please run 'ollama serve' and then 'ollama pull {MODEL_ID}' before using this model."
        )

    return ChatOllama(
        model=MODEL_ID,
        temperature=TEMPERATURE,
        num_predict=MAX_TOKENS,
    )
