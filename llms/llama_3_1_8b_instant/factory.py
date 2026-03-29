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

from langchain_groq import ChatGroq

from .config import (
    API_KEY_ENV,
    MAX_TOKENS,
    MODEL_ID,
    TEMPERATURE,
    WEBSITE,
)


def build() -> ChatGroq:
    """
    Instantiate and return a ChatGroq client for Llama 3.1 8B Instant.

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

    return ChatGroq(
        model=MODEL_ID,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        api_key=api_key,
    )
