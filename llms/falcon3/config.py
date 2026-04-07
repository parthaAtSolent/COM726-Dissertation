from __future__ import annotations

MODEL_KEY:    str = "falcon3"
MODEL_ID:     str = "falcon3:latest"
DISPLAY_NAME: str = "🦅 Falcon 3 (Research & Science)"
ICON:         str = "🦅"

PROVIDER:    str = "ollama"
API_KEY_ENV: str = ""
WEBSITE:     str = "ollama.com"

TEMPERATURE: float = 0.7
MAX_TOKENS:  int = 4096

# Streaming settings
STREAMING_CONFIG = {
    "enabled": True,
    "chunk_size": 5,  # Minimum characters before yielding
    "buffer_timeout_ms": 50,  # Max time to buffer
    "word_mode": True,  # Stream by word boundaries
}
