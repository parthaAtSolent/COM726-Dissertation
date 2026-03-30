"""
langgraph_chatbot/config/settings.py
──────────────────────────────────────
Single source of truth for all application-level settings.

Model configuration lives in llms/ — this file only holds
app-level constants and resolves environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# ── Paths ─────────────────────────────────────────────────────────────────────
# BASE_DIR = COM726/langgraph_chatbot/
BASE_DIR: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = BASE_DIR / "data"
STATIC_DIR: Path = BASE_DIR / "static"
TEMPLATES_DIR: Path = BASE_DIR / "templates"

DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Environment (.env lives at COM726/ root) ──────────────────────────────────
load_dotenv(BASE_DIR.parent / ".env")

DATABASE_PATH: str = os.getenv(
    "DATABASE_PATH", str(DATA_DIR / "chatbot.db")
)

# ── Chat defaults ──────────────────────────────────────────────────────────────
DEFAULT_THREAD_TITLE: str = "New Chat"
MAX_TITLE_LENGTH: int = 50
TITLE_PROMPT_MAX_CHARS: int = 100

# ── Default model key (must match a key in llms/REGISTRY) ─────────────────────
DEFAULT_MODEL_KEY: str = "llama-8b-instant"
