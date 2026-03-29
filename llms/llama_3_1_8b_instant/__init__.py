from __future__ import annotations
from .config import MODEL_KEY, DISPLAY_NAME, ICON
from .factory import build

import llms.llama_3_1_8b_instant as llama_8b

REGISTRY: dict[str, object] = {
    llama_8b.MODEL_KEY: llama_8b,
}


def get_model_package(model_key: str):
    if model_key not in REGISTRY:
        raise KeyError(
            f"Model '{model_key}' is not registered. "
            f"Available: {list(REGISTRY.keys())}"
        )
    return REGISTRY[model_key]


def build_llm(model_key: str):
    return get_model_package(model_key).build()


def get_display_name(model_key: str) -> str:
    pkg = REGISTRY.get(model_key)
    return pkg.DISPLAY_NAME if pkg else model_key


def get_icon(model_key: str) -> str:
    pkg = REGISTRY.get(model_key)
    return pkg.ICON if pkg else "🤖"


def list_model_keys() -> list[str]:
    return list(REGISTRY.keys())
