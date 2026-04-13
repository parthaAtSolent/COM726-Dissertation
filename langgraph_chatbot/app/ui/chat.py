from __future__ import annotations
import time
import traceback
import streamlit as st
import llms
from langchain_core.messages import HumanMessage, AIMessage
from app.core.graph import chatbot
from app.utils import (
    generate_title,
    get_thread_model,
    get_thread_title,
    update_title,
    save_message,
)
from pathlib import Path


# Animation frames (kept in Python for fallback)
_FRAMES = [
    "👩🏻‍🍳 Bro's cooking. Let him cook 🔥",
    "👩🏻‍🍳👩🏻‍🍳 Bro's cooking. Let him cook 🔥🔥",
    "👩🏻‍🍳👩🏻‍🍳👩🏻‍🍳 Bro's cooking. Let him cook 🔥🔥🔥",
]


# ══════════════════════════════════════════════════════════════════════════════
# Helper functions
# ══════════════════════════════════════════════════════════════════════════════

def load_html_template(template_name: str) -> str:
    """Load an HTML template from the templates folder."""
    template_path = Path(__file__).parent.parent / "templates" / template_name
    if template_path.exists():
        return template_path.read_text(encoding="utf-8")
    return ""


# ══════════════════════════════════════════════════════════════════════════════
# Header
# ══════════════════════════════════════════════════════════════════════════════

def render_header() -> None:
    try:
        tid = st.session_state.get("thread_id")
        if not tid:
            return
        title = get_thread_title(tid)
        model_key = get_thread_model(tid)
        st.title(f"💬 {title}")
        st.caption(
            f"Using: {llms.get_icon(model_key)} "
            f"{llms.get_display_name(model_key)}"
        )
    except Exception as e:
        st.error(f"Failed to render header: {str(e)}")


# ══════════════════════════════════════════════════════════════════════════════
# History
# ══════════════════════════════════════════════════════════════════════════════

def render_history() -> None:
    try:
        for msg in st.session_state.get("message_history", []):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
    except Exception as e:
        st.error(f"Failed to render history: {str(e)}")


# ══════════════════════════════════════════════════════════════════════════════
# Streaming — collects full response then returns it
# ══════════════════════════════════════════════════════════════════════════════

def get_full_response(user_input: str, model_key: str, tid: str) -> str:
    """
    Runs the LangGraph stream and returns the final complete response.
    This blocks until the model finishes — animation runs before this call.
    """
    config = {"configurable": {"thread_id": tid}}
    inputs = {
        "messages": [HumanMessage(content=user_input)],
        "model": model_key,
    }

    ai_response = ""

    try:
        for chunk in chatbot.stream(inputs, config=config, stream_mode="values"):
            msgs = chunk.get("messages", [])
            if msgs:
                last = msgs[-1]
                if isinstance(last, AIMessage) and last.content:
                    ai_response = last.content

    except Exception as stream_error:
        error_msg = str(stream_error)
        if "Packet sequence number wrong" in error_msg:
            return "⚠️ Database connection issue. Please refresh and try again."
        elif "BrokenPipeError" in error_msg or "ConnectionError" in error_msg:
            return "⚠️ Connection lost. Please refresh and try again."
        else:
            print(f"Stream error: {stream_error}\n{traceback.format_exc()}")
            return f"⚠️ Error: {str(stream_error)}"

    return ai_response or "⚠️ No response received. Please try again."


# ══════════════════════════════════════════════════════════════════════════════
# Main chat page
# ══════════════════════════════════════════════════════════════════════════════

def render_chat_page() -> None:

    if "message_history" not in st.session_state:
        st.session_state["message_history"] = []
    if "processing_message" not in st.session_state:
        st.session_state["processing_message"] = False

    render_header()

    with st.container():
        render_history()

    user_input = st.chat_input("Type your message here...")

    # ── Step 1: user submits → save + rerun to show message immediately ───────
    if user_input and not st.session_state["processing_message"]:
        tid = st.session_state.get("thread_id")
        if not tid:
            st.error("No thread selected")
            return

        st.session_state["processing_message"] = True
        st.session_state["message_history"].append({
            "role": "user",
            "content": user_input,
        })

        try:
            save_message(tid, "user", user_input)
        except Exception as e:
            print(f"[Chat] Failed to save user message: {e}")

        st.rerun()

    # ── Step 2: processing=True → show animation then get response ───────────
    if st.session_state.get("processing_message", False):
        history = st.session_state.get("message_history", [])

        if not history or history[-1]["role"] != "user":
            st.session_state["processing_message"] = False
            return

        user_message = history[-1]["content"]
        tid = st.session_state.get("thread_id")
        model_key = get_thread_model(tid)

        # Pick animation colour based on current theme (dark=light-purple, light=deep-purple)
        anim_color = "#5a52e0" if st.session_state.get(
            "theme") == "light" else "#c4c0ff"

        with st.chat_message("assistant"):
            placeholder = st.empty()

            # ── Animate on the main thread while model runs in background ─────
            import concurrent.futures

            full_response = ""
            frame = 0

            # Submit model call to thread pool
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    get_full_response, user_message, model_key, tid
                )

                # Cycle animation frames until model finishes
                while not future.done():
                    placeholder.markdown(
                        f"<span style='color:{anim_color};font-style:italic;"
                        f"font-size:1rem;font-weight:500'>"
                        f"{_FRAMES[frame % len(_FRAMES)]}</span>",
                        unsafe_allow_html=True,
                    )
                    frame += 1
                    time.sleep(0.5)

                # Get result
                try:
                    full_response = future.result()
                except Exception as e:
                    full_response = f"⚠️ Error: {e}"

            # ── Show final response ───────────────────────────────────────────
            placeholder.markdown(full_response)

        # ── Save + update state ───────────────────────────────────────────────
        if full_response and not full_response.startswith("⚠️"):
            st.session_state["message_history"].append({
                "role": "assistant",
                "content": full_response,
            })

            try:
                save_message(tid, "assistant", full_response)
            except Exception as e:
                print(f"[Chat] Failed to save assistant message: {e}")

            # Auto-title on first exchange
            if len(st.session_state["message_history"]) == 2:
                try:
                    new_title = generate_title(user_message, model_key)
                    update_title(tid, new_title)
                    st.toast(f"Chat titled: {new_title}", icon="✨")
                except Exception as title_error:
                    print(f"Title generation error: {title_error}")

        st.session_state["processing_message"] = False
        st.rerun()


if __name__ == "__main__":
    render_chat_page()
