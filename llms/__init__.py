"""
llms/__init__.py
─────────────────
Central LLM registry.

How to add a new model
──────────────────────
1. Create  llms/<your_model_folder>/
2. Add     config.py  (MODEL_KEY, DISPLAY_NAME, ICON, …)
3. Add     factory.py (def build() -> <LangChain chat model>)
4. Add     __init__.py with: from .config import MODEL_KEY, DISPLAY_NAME, ICON
                              from .factory import build
5. Import the new package in the REGISTRY dict below — done.
"""

from __future__ import annotations

from llms.llama_3_1_8b_instant import MODEL_KEY, DISPLAY_NAME, ICON, build as _build_llama_8b

REGISTRY: dict[str, dict] = {
    MODEL_KEY: {
        "name": DISPLAY_NAME,
        "icon": ICON,
        "build": _build_llama_8b,
    },
    # ── Add future models below ────────────────────────────────────────────
    # gemini_flash.MODEL_KEY: {
    #     "name": gemini_flash.DISPLAY_NAME,
    #     "icon": gemini_flash.ICON,
    #     "build": gemini_flash.build,
    # },
}


def build_llm(model_key: str):
    """Build and return the LangChain chat model for *model_key*."""
    if model_key not in REGISTRY:
        raise KeyError(
            f"Model '{model_key}' not registered. "
            f"Available: {list(REGISTRY.keys())}"
        )
    return REGISTRY[model_key]["build"]()


def list_model_keys() -> list[str]:
    """Return all registered model keys in display order."""
    return list(REGISTRY.keys())


def get_display_name(model_key: str) -> str:
    """Return the human-readable display name for *model_key*."""
    info = REGISTRY.get(model_key)
    return info["name"] if info else model_key


def get_icon(model_key: str) -> str:
    """Return the emoji icon for *model_key*."""
    info = REGISTRY.get(model_key)
    return info["icon"] if info else "🤖"
