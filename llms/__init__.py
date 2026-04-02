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

from llms.llama_3_1_8b_instant.config import MODEL_KEY as _k1, DISPLAY_NAME as _n1, ICON as _i1
from llms.llama_3_1_8b_instant.factory import build as _build_llama_8b

from llms.gemini_2_5_flash.config import MODEL_KEY as _k2, DISPLAY_NAME as _n2, ICON as _i2
from llms.gemini_2_5_flash.factory import build as _build_gemini

from llms.qwen3_5_0_8b.config import MODEL_KEY as _k4, DISPLAY_NAME as _n4, ICON as _i4
from llms.qwen3_5_0_8b.factory import build as _build_qwen

from llms.phi3_3_8b.config import MODEL_KEY as _k5, DISPLAY_NAME as _n5, ICON as _i5
from llms.phi3_3_8b.factory import build as _build_phi3

from llms.granite3_dense_2b.config import MODEL_KEY as _k6, DISPLAY_NAME as _n6, ICON as _i6
from llms.granite3_dense_2b.factory import build as _build_granite3

from llms.gemma3_270m.config import MODEL_KEY as _k7, DISPLAY_NAME as _n7, ICON as _i7
from llms.gemma3_270m.factory import build as _build_gemma3


REGISTRY: dict[str, dict] = {
    _k1: {"name": _n1, "icon": _i1, "build": _build_llama_8b},
    _k2: {"name": _n2, "icon": _i2, "build": _build_gemini},
    _k4: {"name": _n4, "icon": _i4, "build": _build_qwen},
    _k5: {"name": _n5, "icon": _i5, "build": _build_phi3},
    _k6: {"name": _n6, "icon": _i6, "build": _build_granite3},
    _k7: {"name": _n7, "icon": _i7, "build": _build_gemma3},
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
