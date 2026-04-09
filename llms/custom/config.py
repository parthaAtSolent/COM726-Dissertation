"""
llms/custom/config.py
──────────────────────
Configuration settings for the custom multi-model orchestrator.

These settings control how the orchestrator appears in the UI
and default generation parameters.
"""

# Model identification (used by the UI/registry)
MODEL_KEY: str = "custom"           # Internal unique identifier
MODEL_ID: str = "custom"            # Alternative identifier
DISPLAY_NAME: str = "🎯 Auto (Multi-Model Routing)"  # User-facing name
ICON: str = "🎯"                     # Emoji icon for UI

# Provider information
PROVIDER: str = "custom"            # Provider name
# Environment variable for API key (if needed)
API_KEY_ENV: str = ""
WEBSITE: str = ""                   # Provider website (optional)

# Default generation parameters
# Controls randomness (0=deterministic, 1=creative)
TEMPERATURE: float = 0.7
MAX_TOKENS: int = 4096              # Maximum response length
