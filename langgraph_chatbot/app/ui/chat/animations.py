"""Animation frames for chat responses."""

# Standard response animation frames
STANDARD_FRAMES = [
    "👩🏻‍🍳 Bro's cooking. Let him cook 🔥",
    "👩🏻‍🍳👩🏻‍🍳 Bro's cooking. Let him cook 🔥🔥",
    "👩🏻‍🍳👩🏻‍🍳👩🏻‍🍳 Bro's cooking. Let him cook 🔥🔥🔥",
]

# RAG‑specific animation frames
RAG_FRAMES = [
    "📚 Searching documents... 🔍",
    "📄 Reading uploaded files... 📖",
    "🤔 Finding relevant context... ✨",
]


def get_animation_frames(has_rag: bool = False) -> list:
    """Get animation frames based on whether RAG is active."""
    return RAG_FRAMES if has_rag else STANDARD_FRAMES
