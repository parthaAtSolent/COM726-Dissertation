from __future__ import annotations

MODEL_KEY: str = "granite3-dense-2b"
MODEL_ID: str = "granite3-dense:2b"
DISPLAY_NAME: str = "💎 Granite 3 Dense 2B (Local)"
ICON: str = "💎"

PROVIDER: str = "ollama"
API_KEY_ENV: str = ""
WEBSITE: str = "ollama.com"

TEMPERATURE: float = 0.7
MAX_TOKENS: int = 4096

# Streaming settings
STREAMING_CONFIG = {
    "enabled": True,
    "chunk_size": 5,  # Minimum characters before yielding
    "buffer_timeout_ms": 50,  # Max time to buffer
    "word_mode": True,  # Stream by word boundaries
}
