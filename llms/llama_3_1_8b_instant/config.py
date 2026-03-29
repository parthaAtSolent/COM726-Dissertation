"""
llms/llama_3_1_8b_instant/config.py
─────────────────────────────────────
All configuration specific to the Llama 3.1 8B Instant model.

To add a new model later, create a sibling folder (e.g. llms/gemini_2_5_flash/)
and replicate this file with the appropriate values.  No other file in the
project needs to change.
"""

# ── Identity ──────────────────────────────────────────────────────────────────
MODEL_KEY: str = "llama-8b-instant"
MODEL_ID: str = "llama-3.1-8b-instant"          # Groq model string
DISPLAY_NAME: str = "⚡ Llama 3.1 8B Instant (Fast)"
ICON: str = "⚡"

# ── Provider ──────────────────────────────────────────────────────────────────
PROVIDER: str = "groq"
API_KEY_ENV: str = "GROQ_API_KEY"               # env var name in .env
WEBSITE: str = "console.groq.com"

# ── Generation parameters ─────────────────────────────────────────────────────
TEMPERATURE: float = 0.7
MAX_TOKENS: int = 4096
