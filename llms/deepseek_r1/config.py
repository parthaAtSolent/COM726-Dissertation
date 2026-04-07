MODEL_KEY:    str = "deepseek-r1"
MODEL_ID:     str = "deepseek-r1:latest"
DISPLAY_NAME: str = "🤔 DeepSeek R1 (Reasoning)"
ICON:         str = "🤔"

PROVIDER:    str = "ollama"
API_KEY_ENV: str = ""
WEBSITE:     str = "ollama.com"

TEMPERATURE: float = 0.6
MAX_TOKENS:  int = 4096

# Streaming settings
STREAMING_CONFIG = {
    "enabled": True,
    "chunk_size": 5,  # Minimum characters before yielding
    "show_thinking": True,  # Show DeepSeek reasoning
    "buffer_timeout_ms": 50,  # Max time to buffer
    "word_mode": True,  # Stream by word boundaries
}
