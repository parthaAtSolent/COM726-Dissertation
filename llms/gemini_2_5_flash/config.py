from __future__ import annotations

MODEL_KEY: str = "gemini-2.5-flash"
MODEL_ID: str = "gemini-2.5-flash"
DISPLAY_NAME: str = "⚡ Gemini 2.5 Flash"
ICON: str = "⚡"

PROVIDER: str = "google"
API_KEY_ENV: str = "GOOGLE_API_KEY"
WEBSITE: str = "aistudio.google.com"

TEMPERATURE: float = 0.7
MAX_TOKENS: int = 4096

# Streaming settings
STREAMING_CONFIG = {
    "enabled": True,
    "chunk_size": 5,  # Minimum characters before yielding
    "buffer_timeout_ms": 50,  # Max time to buffer
    "word_mode": True,  # Stream by word boundaries
}
