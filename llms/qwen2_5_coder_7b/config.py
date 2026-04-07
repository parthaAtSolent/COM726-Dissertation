"""
Qwen2.5-Coder 7B Configuration
Specialized for code generation, explanation, and conversation.
"""

MODEL_KEY:    str = "qwen2.5-coder-7b"
MODEL_ID:     str = "qwen2.5-coder:7b"
DISPLAY_NAME: str = "💻 Qwen2.5 Coder 7B"
ICON:         str = "💻"

PROVIDER:    str = "ollama"
API_KEY_ENV: str = ""
WEBSITE:     str = "ollama.com/library/qwen2.5-coder"

# Optimal settings for code tasks
TEMPERATURE: float = 0.3    # Lower for precise code, higher for explanations
MAX_TOKENS:  int = 8192    # Enough for most code blocks

# Optional: Add system prompt for code tasks
SYSTEM_PROMPT = """
You are an expert coding assistant.
Always provide complete responses.
If the user asks for multiple code examples, provide all of them.
If the user asks for a comparison, include a markdown table.
Never stop after the first code block.
"""
