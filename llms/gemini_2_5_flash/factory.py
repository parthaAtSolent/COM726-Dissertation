from __future__ import annotations
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from .config import API_KEY_ENV, MAX_TOKENS, MODEL_ID, TEMPERATURE, WEBSITE


def build() -> ChatGoogleGenerativeAI:
    api_key = os.getenv(API_KEY_ENV, "")
    if not api_key:
        raise EnvironmentError(
            f"'{API_KEY_ENV}' is not set in your .env file. "
            f"Get a free key at https://{WEBSITE}"
        )

    try:
        llm = ChatGoogleGenerativeAI(
            model=MODEL_ID,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            google_api_key=api_key,
        )
        # Lightweight connectivity check — invoke with an empty prompt
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
