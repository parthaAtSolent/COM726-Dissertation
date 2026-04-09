"""
llms/custom/__init__.py
────────────────────────
Module initializer for the custom multi-model orchestrator.

This module provides a LangChain-compatible chat model that intelligently
routes requests to specialist models based on task type and complexity.
"""

# Import the main components for easy access
from .config import MODEL_KEY, DISPLAY_NAME, ICON
from .factory import build, CustomOrchestrator

# Define what gets exported when someone does "from llms.custom import *"
__all__ = [
    'MODEL_KEY',           # Unique identifier for this model
    'DISPLAY_NAME',        # Human-readable name for UI
    'ICON',                # Icon for UI display
    'build',               # Factory function to create orchestrator
    'CustomOrchestrator',  # Main orchestrator class
]

# Optional: Print initialization message (useful for debugging)
# Uncomment the following line if you want to see when this module loads
# print(f"[llms.custom] Initialized multi-model orchestrator: {DISPLAY_NAME}")
