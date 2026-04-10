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
    save_message,  # ADDED: Import save_message
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
# Streaming response generator
# ══════════════════════════════════════════════════════════════════════════════

def stream_response(user_input: str, model_key: str, tid: str):
    """Generator function to stream the AI response."""
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
                    yield ai_response

    except Exception as stream_error:
        error_msg = str(stream_error)
        if "Packet sequence number wrong" in error_msg:
            ai_response = "⚠️ Database connection issue. Please refresh and try again."
        elif "BrokenPipeError" in error_msg or "ConnectionError" in error_msg:
            ai_response = "⚠️ Connection lost. Please refresh and try again."
        else:
            print(f"Stream error: {stream_error}\n{traceback.format_exc()}")
            ai_response = f"⚠️ Error: {str(stream_error)}"
        yield ai_response

    if not ai_response:
        ai_response = "⚠️ No response received. Please try again."
        yield ai_response


# ══════════════════════════════════════════════════════════════════════════════
# Main Chat Page Render Function
# ══════════════════════════════════════════════════════════════════════════════

def render_chat_page():
    """Main function to render the chat page."""

    # Initialize session state
    if "message_history" not in st.session_state:
        st.session_state["message_history"] = []

    if "processing_message" not in st.session_state:
        st.session_state["processing_message"] = False

    # Render header
    render_header()

    # Create a container for the chat history
    chat_container = st.container()

    # Render existing history in the container
    with chat_container:
        render_history()

    # Chat input
    user_input = st.chat_input("Type your message here...")

    # Process the input if it exists and we're not already processing
    if user_input and not st.session_state["processing_message"]:
        tid = st.session_state.get("thread_id")
        if not tid:
            st.error("No thread selected")
            return

        model_key = get_thread_model(tid)

        # Set processing flag
        st.session_state["processing_message"] = True

        # Add user message to history
        st.session_state["message_history"].append({
            "role": "user",
            "content": user_input,
        })

        # Save user message to MySQL
        try:
            save_message(tid, "user", user_input)
            print(f"[Chat] Saved user message to MySQL for thread {tid}")
        except Exception as e:
            print(f"[Chat] Failed to save user message: {e}")

        # Rerun to show the user message immediately
        st.rerun()

    # If we're processing a message, handle the streaming response
    if st.session_state.get("processing_message", False):
        # Get the last user message
        if st.session_state["message_history"] and st.session_state["message_history"][-1]["role"] == "user":
            user_message = st.session_state["message_history"][-1]["content"]
            tid = st.session_state.get("thread_id")
            model_key = get_thread_model(tid)

            # Create assistant message placeholder
            with st.chat_message("assistant"):
                # Show cooking message
                cooking_placeholder = st.empty()
                cooking_placeholder.markdown(
                    "Bro's cooking... Let him cook 🔥🔥🔥")

                # Stream the response
                response_placeholder = st.empty()
                full_response = ""

                try:
                    # Stream the response
                    for chunk in stream_response(user_message, model_key, tid):
                        full_response = chunk
                        response_placeholder.markdown(full_response + "▌")

                    # Final response without cursor
                    if full_response:
                        response_placeholder.markdown(full_response)

                        # Save successful response to history
                        if not full_response.startswith("⚠️"):
                            st.session_state["message_history"].append({
                                "role": "assistant",
                                "content": full_response,
                            })

                            # Save assistant message to MySQL
                            try:
                                save_message(tid, "assistant", full_response)
                                print(
                                    f"[Chat] Saved assistant message to MySQL for thread {tid}")
                            except Exception as e:
                                print(
                                    f"[Chat] Failed to save assistant message: {e}")

                            # Auto-generate title on first exchange
                            if len(st.session_state["message_history"]) == 2:
                                try:
                                    new_title = generate_title(
                                        user_message, model_key)
                                    update_title(tid, new_title)
                                    st.toast(
                                        f"Chat titled: {new_title}", icon="✨")
                                except Exception as title_error:
                                    print(
                                        f"Title generation error: {title_error}")

                except Exception as e:
                    st.error(f"⚠️ Error: {e}")
                    print(f"Error in streaming: {e}\n{traceback.format_exc()}")

                finally:
                    # Clear processing flag
                    st.session_state["processing_message"] = False
                    # Rerun to update the UI with the final state
                    st.rerun()


# This would be called from your main app.py file
if __name__ == "__main__":
    render_chat_page()
