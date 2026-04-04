from __future__ import annotations
from app.ui import render_sidebar, render_header, render_history, handle_input
from app.utils.thread_service import init_threads_table
from app.utils import (
    create_thread,
    inject_css,
    inject_js,
    list_threads,
    load_conversation,
    new_thread_id,
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
inject_js("utils.js")
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
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    try:
        _bootstrap()
        render_sidebar()
        render_header()
        render_history()

        user_input = st.chat_input("Type your message here…")
        if user_input:
            handle_input(user_input)

    except Exception as e:
        st.error(f"Application error: {str(e)}")
        print(f"Main error: {traceback.format_exc()}")
        st.stop()


if __name__ == "__main__":
    main()
