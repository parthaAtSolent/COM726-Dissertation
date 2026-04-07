"""
Qwen2.5-Coder Factory
Builds the LangChain chat model for Qwen2.5-Coder.
"""

from __future__ import annotations
import httpx
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage
from .config import MODEL_ID, TEMPERATURE, MAX_TOKENS, SYSTEM_PROMPT


def build() -> ChatOllama:
    """
    Build Qwen2.5-Coder 7B model.
    Specialized for code generation, explanation, and conversation about code.
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

    model = ChatOllama(
        model=MODEL_ID,
        temperature=TEMPERATURE,
        num_predict=MAX_TOKENS,
        stop=["User:", "Human:"],
    )

    return model


def build_with_system_prompt() -> ChatOllama:
    """
    Build Qwen2.5-Coder with a system prompt (for chat applications).
    """
    model = build()

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

        def __getattr__(self, name):
            return getattr(self.model, name)

    return SystemPromptWrapper(model, SYSTEM_PROMPT)
