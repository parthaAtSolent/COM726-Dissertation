from __future__ import annotations
from app.ui import render_sidebar, render_header, render_history, handle_input
from app.utils.thread_service import (
    init_threads_table,
    generate_response_with_context,
    save_message,
    create_thread,
    list_threads,
    load_conversation,
    new_thread_id,
)
from app.utils import (
    inject_css,
    inject_js,
)
import llms
import streamlit as st
import traceback
import sys
from pathlib import Path

# ── Path fix ──────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LangGraph Chatbot · COM726",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


inject_css("sidebar.css", "chat.css")
inject_js("theme.js", "utils.js")

init_threads_table()


# ══════════════════════════════════════════════════════════════════════════════
# Bootstrap
# ══════════════════════════════════════════════════════════════════════════════

def _bootstrap() -> None:
    if st.session_state.get("_initialized"):
        return
    try:
        model_keys = llms.list_model_keys()
        default_model = model_keys[0] if model_keys else "llama-8b-instant"
        existing = list_threads()

        if not existing:
            tid = new_thread_id()
            create_thread(tid, title="New Chat", model=default_model)
            st.session_state["thread_id"] = tid
            st.session_state["message_history"] = []
        else:
            first = existing[0]
            tid = first["thread_id"]
            default_model = first.get("model", default_model)
            st.session_state["thread_id"] = tid
            st.session_state["message_history"] = load_conversation(tid)

        st.session_state.update({
            "selected_model":     default_model,
            "editing_thread":     None,
            "delete_confirm":     None,
            "processing_message": False,
            "_initialized":       True,
        })
    except Exception as e:
        st.error(f"Failed to initialize application: {str(e)}")
        st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# Streaming response handler
# ══════════════════════════════════════════════════════════════════════════════

def save_message_to_db(thread_id: str, role: str, content: str):
    """Save a message to the database."""
    save_message(thread_id, role, content)


def process_message_with_streaming(user_input: str):
    """Process user message with streaming response."""

    # Add user message to history
    st.session_state.message_history.append(
        {"role": "user", "content": user_input})
    save_message_to_db(st.session_state.thread_id, "user", user_input)

    # Create placeholder for streaming response
    with st.chat_message("assistant"):
        placeholder = st.empty()

        try:
            # Check if selected model supports streaming
            model_key = st.session_state.selected_model

            # Use streaming response generation
            full_response = generate_response_with_context(
                model_key=model_key,
                conversation_history=st.session_state.message_history,
                user_message=user_input,
                placeholder=placeholder,
                show_thinking=True  # Show DeepSeek reasoning steps
            )

            # Save assistant response
            st.session_state.message_history.append(
                {"role": "assistant", "content": full_response})
            save_message_to_db(st.session_state.thread_id,
                               "assistant", full_response)

        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            placeholder.error(error_msg)
            st.session_state.message_history.append(
                {"role": "assistant", "content": error_msg})
            save_message_to_db(st.session_state.thread_id,
                               "assistant", error_msg)

    st.session_state.processing_message = False
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    try:
        _bootstrap()
        render_sidebar()
        render_header()
        render_history()

        user_input = st.chat_input("Type your message here…")
        if user_input and not st.session_state.get("processing_message", False):
            st.session_state.processing_message = True
            process_message_with_streaming(user_input)

    except Exception as e:
        st.error(f"Application error: {str(e)}")
        print(f"Main error: {traceback.format_exc()}")
        st.stop()


if __name__ == "__main__":
    main()
