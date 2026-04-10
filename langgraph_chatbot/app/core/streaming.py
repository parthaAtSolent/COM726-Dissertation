"""
app/core/streaming.py
─────────────────────
Streaming response handling.
"""

from __future__ import annotations

import traceback
import streamlit as st
from app.utils.thread_service import (
    generate_response_with_context,
    save_message,
)


def save_message_to_db(thread_id: str, role: str, content: str) -> bool:
    """Save a message to the database."""
    try:
        save_message(thread_id, role, content)
        return True
    except Exception as e:
        print(f"Failed to save message: {e}")
        return False


def process_pending_message() -> None:
    """Process pending user message with streaming response."""
    user_message = st.session_state.get("pending_user_input")

    if not user_message:
        return

    st.session_state.pending_user_input = None

    with st.chat_message("assistant"):
        cooking_placeholder = st.empty()
        cooking_placeholder.markdown("Bro's cooking... Let him cook 🔥🔥🔥")
        response_placeholder = st.empty()

        try:
            model_key = st.session_state.selected_model

            full_response = generate_response_with_context(
                model_key=model_key,
                conversation_history=st.session_state.message_history[:-1],
                user_message=user_message,
                placeholder=response_placeholder,
                show_thinking=True
            )

            cooking_placeholder.empty()
            response_placeholder.markdown(full_response)

            st.session_state.message_history.append(
                {"role": "assistant", "content": full_response}
            )
            save_message_to_db(st.session_state.thread_id,
                               "assistant", full_response)

        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            cooking_placeholder.empty()
            response_placeholder.error(error_msg)
            st.session_state.message_history.append(
                {"role": "assistant", "content": error_msg}
            )
            save_message_to_db(st.session_state.thread_id,
                               "assistant", error_msg)
            print(f"Streaming error: {traceback.format_exc()}")

        finally:
            st.session_state.processing_message = False
