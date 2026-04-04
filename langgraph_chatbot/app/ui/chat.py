from __future__ import annotations
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
)


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
# Input handler
# ══════════════════════════════════════════════════════════════════════════════

def handle_input(user_input: str) -> None:
    if not user_input or st.session_state.get("processing_message"):
        return

    tid = st.session_state["thread_id"]
    model_key = get_thread_model(tid)

    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state["message_history"].append({
        "role":    "user",
        "content": user_input,
    })
    st.session_state["processing_message"] = True

    ai_response = ""

    try:
        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("▌")

            config = {"configurable": {"thread_id": tid}}
            inputs = {
                "messages": [HumanMessage(content=user_input)],
                "model":    model_key,
            }

            try:
                for chunk in chatbot.stream(inputs, config=config, stream_mode="values"):
                    msgs = chunk.get("messages", [])
                    if msgs:
                        last = msgs[-1]
                        if isinstance(last, AIMessage) and last.content:
                            ai_response = last.content
                            placeholder.markdown(ai_response + "▌")
            except Exception as stream_error:
                error_msg = str(stream_error)
                if "Packet sequence number wrong" in error_msg:
                    ai_response = "⚠️ Database connection issue. Please refresh and try again."
                elif "BrokenPipeError" in error_msg or "ConnectionError" in error_msg:
                    ai_response = "⚠️ Connection lost. Please refresh and try again."
                else:
                    print(
                        f"Stream error: {stream_error}\n{traceback.format_exc()}")
                    ai_response = f"⚠️ Error: {str(stream_error)}"
                placeholder.markdown(ai_response)

            if not ai_response:
                ai_response = "⚠️ No response received. Please try again."
                placeholder.markdown(ai_response)

        if ai_response and not ai_response.startswith("⚠️"):
            st.session_state["message_history"].append({
                "role":    "assistant",
                "content": ai_response,
            })
            # Auto-generate title on first exchange
            if len(st.session_state["message_history"]) == 2:
                try:
                    new_title = generate_title(user_input, model_key)
                    update_title(tid, new_title)
                    st.toast(f"Chat titled: {new_title}", icon="✨")
                except Exception as title_error:
                    print(f"Title generation error: {title_error}")

    except Exception as exc:
        st.error(f"⚠️ Error: {exc}")
        print(f"Error in handle_input: {exc}\n{traceback.format_exc()}")
        if st.session_state["message_history"]:
            st.session_state["message_history"].pop()
    finally:
        st.session_state["processing_message"] = False
        st.rerun()
