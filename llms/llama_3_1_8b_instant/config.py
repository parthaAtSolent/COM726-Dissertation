"""
llms/llama_3_1_8b_instant/config.py
─────────────────────────────────────
All configuration specific to the Llama 3.1 8B Instant model.
"""

from __future__ import annotations

# ── Identity ──────────────────────────────────────────────────────────────────
MODEL_KEY: str = "llama-8b-instant"
MODEL_ID: str = "llama3.1:8b"                  # Local Ollama model name
DISPLAY_NAME: str = "🏃🏻 Llama 3.1 8B Instant (Local)"
ICON: str = "🏃🏻"

# ── Provider ──────────────────────────────────────────────────────────────────
PROVIDER: str = "ollama"                       # Changed from groq to ollama
API_KEY_ENV: str = ""                          # No API key needed for local
WEBSITE: str = "localhost:11434"               # Default Ollama port

# ── Generation parameters ─────────────────────────────────────────────────────
TEMPERATURE: float = 0.7
MAX_TOKENS: int = 4096

# Streaming settings
STREAMING_CONFIG = {
    "enabled": True,
    "chunk_size": 5,
    "buffer_timeout_ms": 50,
    "word_mode": True,
}

# Ollama specific settings
OLLAMA_BASE_URL: str = "http://localhost:11434"
