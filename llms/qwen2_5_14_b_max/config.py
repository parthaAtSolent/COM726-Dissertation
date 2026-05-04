from __future__ import annotations

MODEL_KEY: str = "qwen2.5-14b-max"
MODEL_ID: str = "m/qwen2514bmax:latest"          # Ollama tag for Qwen2.5 14B Max
DISPLAY_NAME: str = "🐉 Qwen 2.5 14B Max (Local)"
ICON: str = "🐲"

PROVIDER: str = "ollama"
API_KEY_ENV: str = ""
WEBSITE: str = "ollama.com"

TEMPERATURE: float = 0.7
MAX_TOKENS: int = 2048

# Streaming settings
STREAMING_CONFIG = {
    "enabled": True,
    "chunk_size": 5,  # Minimum characters before yielding
    "buffer_timeout_ms": 50,  # Max time to buffer
    "word_mode": True,  # Stream by word boundaries
}