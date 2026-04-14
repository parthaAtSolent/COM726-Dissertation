"""Chat message processing and response handling."""

import time
import traceback
import streamlit as st
import llms
from langchain_core.messages import HumanMessage, AIMessage
from app.core.graph import chatbot
from app.utils import (
    generate_title,
    get_thread_model,
    update_title,
    save_message,
)
from app.rag import list_ingested_files
from app.ui.chat.animations import get_animation_frames
from app.ui.chat.history import render_history


def render_chat_page() -> None:
    """Render the main chat page with message handling."""
    if "message_history" not in st.session_state:
        st.session_state["message_history"] = []
    if "processing_message" not in st.session_state:
        st.session_state["processing_message"] = False

    from app.ui.chat.header import render_header
    render_header()

    with st.container():
        render_history()

    user_input = st.chat_input("Type your message here...")

    if user_input and not st.session_state["processing_message"]:
        _handle_user_input(user_input)

    if st.session_state.get("processing_message", False):
        _process_assistant_response()


def _handle_user_input(user_input: str) -> None:
    """Handle user message submission."""
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


def _process_assistant_response() -> None:
    """Process and display assistant response."""
    history = st.session_state.get("message_history", [])

    if not history or history[-1]["role"] != "user":
        st.session_state["processing_message"] = False
        return

    user_message = history[-1]["content"]
    tid = st.session_state.get("thread_id")
    model_key = get_thread_model(tid)

    has_rag = len(list_ingested_files()) > 0
    display_frames = get_animation_frames(has_rag)
    anim_color = "#5a52e0" if st.session_state.get(
        "theme") == "light" else "#c4c0ff"

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = _get_response_with_animation(
            user_message, model_key, tid, placeholder, display_frames, anim_color
        )

    if full_response and not full_response.startswith("⚠️"):
        st.session_state["message_history"].append({
            "role": "assistant",
            "content": full_response,
        })

        try:
            save_message(tid, "assistant", full_response)
        except Exception as e:
            print(f"[Chat] Failed to save assistant message: {e}")

        if len(st.session_state["message_history"]) == 2:
            _generate_auto_title(user_message, model_key, tid)

    st.session_state["processing_message"] = False
    st.rerun()


def _get_response_with_animation(
    user_message: str,
    model_key: str,
    tid: str,
    placeholder,
    frames: list,
    color: str
) -> str:
    """Get response while showing animation."""
    import concurrent.futures

    full_response = ""
    frame = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            _get_full_response, user_message, model_key, tid)

        while not future.done():
            placeholder.markdown(
                f"<span style='color:{color};font-style:italic;"
                f"font-size:1rem;font-weight:500'>"
                f"{frames[frame % len(frames)]}</span>",
                unsafe_allow_html=True,
            )
            frame += 1
            time.sleep(0.5)

        try:
            full_response = future.result()
        except Exception as e:
            full_response = f"⚠️ Error: {e}"

    placeholder.markdown(full_response)
    return full_response


def _get_full_response(user_input: str, model_key: str, tid: str) -> str:
    """Run the LangGraph stream and return the final complete response."""
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


def _generate_auto_title(user_message: str, model_key: str, tid: str) -> None:
    """Generate and save auto-title for new conversations."""
    try:
        new_title = generate_title(user_message, model_key)
        update_title(tid, new_title)
        st.toast(f"Chat titled: {new_title}", icon="✨")
    except Exception as title_error:
        print(f"Title generation error: {title_error}")
