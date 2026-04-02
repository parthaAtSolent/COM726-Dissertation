"""
langgraph_chatbot/config/settings.py
"""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = BASE_DIR / "data"
STATIC_DIR: Path = BASE_DIR / "static"
TEMPLATES_DIR: Path = BASE_DIR / "templates"

DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Environment ───────────────────────────────────────────────────────────────
load_dotenv(BASE_DIR.parent / ".env")
# print(os.getenv("GOOGLE_API_KEY"))
# print(os.getenv("GROQ_API_KEY"))

# ── MySQL ──────────────────────────────────────────────────────────────────────
MYSQL_HOST: str = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT: int = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER: str = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD: str = os.getenv("MYSQL_PASSWORD", "admin123")
MYSQL_DATABASE: str = os.getenv("MYSQL_DATABASE", "langgraph_chatbot")

# ── Chat defaults ──────────────────────────────────────────────────────────────
DEFAULT_THREAD_TITLE: str = "New Chat"
MAX_TITLE_LENGTH: int = 50
TITLE_PROMPT_MAX_CHARS: int = 100
DEFAULT_MODEL_KEY: str = "llama-8b-instant"
