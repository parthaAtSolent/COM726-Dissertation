"""config — application-level settings."""

from .settings import (
    BASE_DIR,
    DATABASE_PATH,
    DATA_DIR,
    DEFAULT_MODEL_KEY,
    DEFAULT_THREAD_TITLE,
    MAX_TITLE_LENGTH,
    STATIC_DIR,
    TEMPLATES_DIR,
    TITLE_PROMPT_MAX_CHARS,
)

__all__ = [
    "BASE_DIR",
    "DATABASE_PATH",
    "DATA_DIR",
    "DEFAULT_MODEL_KEY",
    "DEFAULT_THREAD_TITLE",
    "MAX_TITLE_LENGTH",
    "STATIC_DIR",
    "TEMPLATES_DIR",
    "TITLE_PROMPT_MAX_CHARS",
]
