from __future__ import annotations
from app.utils import (
    create_thread,
    delete_thread,
    generate_title,
    get_most_recent_thread_id,
    get_thread_model,
    get_thread_title,
    inject_css,
    inject_js,
    list_threads,
    load_conversation,
    load_template,
    new_thread_id,
    update_model,
    update_title,
)
from app.core.graph import chatbot
from langchain_core.messages import HumanMessage, AIMessage
import streamlit as st
import llms
import traceback

# ── FIX PATH FIRST ───────────────────────────────────────────────────────────
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent  # points to COM726/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Local imports AFTER path fix ─────────────────────────────────────────────

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LangGraph Chatbot · COM726",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css("sidebar.css", "chat.css")
inject_js("utils.js")

# ══════════════════════════════════════════════════════════════════════════════
# Session-state bootstrap
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

        st.session_state["selected_model"] = default_model
        st.session_state["editing_thread"] = None
        st.session_state["delete_confirm"] = None
        st.session_state["processing_message"] = False
        st.session_state["_initialized"] = True
    except Exception as e:
        st.error(f"Failed to initialize application: {str(e)}")
        st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# Thread actions
# ══════════════════════════════════════════════════════════════════════════════

def _new_chat() -> None:
    try:
        tid = new_thread_id()

        model = st.session_state.get(
            "selected_model",
            llms.list_model_keys()[0] if llms.list_model_keys(
            ) else "llama-8b-instant"
        )

        create_thread(tid, title="New Chat", model=model)

        st.session_state.update({
            "thread_id": tid,
            "message_history": [],
            "editing_thread": None,
            "delete_confirm": None,
        })

        st.rerun()
    except Exception as e:
        st.error(f"Failed to create new chat: {str(e)}")


def _switch_thread(thread_id: str) -> None:
    if st.session_state["thread_id"] == thread_id:
        return

    try:
        st.session_state.update({
            "thread_id": thread_id,
            "message_history": load_conversation(thread_id),
            "selected_model": get_thread_model(thread_id),
            "editing_thread": None,
            "delete_confirm": None,
        })

        st.rerun()
    except Exception as e:
        st.error(f"Failed to switch thread: {str(e)}")


def _delete_thread(thread_id: str) -> None:
    try:
        delete_thread(thread_id)

        if st.session_state["thread_id"] == thread_id:

            recent = get_most_recent_thread_id()

            if recent:
                st.session_state.update({
                    "thread_id": recent,
                    "message_history": load_conversation(recent),
                    "selected_model": get_thread_model(recent),
                })
            else:
                _new_chat()
                return

        st.session_state["delete_confirm"] = None

        st.rerun()
    except Exception as e:
        st.error(f"Failed to delete thread: {str(e)}")


def _save_title(thread_id: str, new_title: str) -> None:
    try:
        if new_title.strip():
            update_title(thread_id, new_title.strip())

        st.session_state["editing_thread"] = None

        st.rerun()
    except Exception as e:
        st.error(f"Failed to save title: {str(e)}")


def _on_model_change() -> None:
    try:
        chosen = st.session_state["model_selector"]

        st.session_state["selected_model"] = chosen

        tid = st.session_state.get("thread_id")

        if tid:
            update_model(tid, chosen)
    except Exception as e:
        st.error(f"Failed to change model: {str(e)}")


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════

def _render_branding() -> None:
    try:
        html = load_template("sidebar.html")

        if html:
            st.sidebar.markdown(html, unsafe_allow_html=True)
    except Exception as e:
        st.sidebar.error(f"Failed to load branding: {str(e)}")


def _render_model_selector() -> None:
    try:
        keys = llms.list_model_keys()

        if not keys:
            st.sidebar.error("No models available")
            return

        display_map = {
            k: llms.get_display_name(k)
            for k in keys
        }

        current = st.session_state.get(
            "selected_model",
            keys[0]
        )

        idx = keys.index(current) if current in keys else 0

        st.sidebar.subheader("🤖 Model")

        st.sidebar.selectbox(
            "Choose AI Model",
            options=keys,
            format_func=lambda k: display_map.get(k, k),
            index=idx,
            key="model_selector",
            on_change=_on_model_change,
        )

        st.sidebar.caption("⚡ Groq free tier.")
        st.sidebar.divider()
    except Exception as e:
        st.sidebar.error(f"Failed to load model selector: {str(e)}")


def _render_thread_row(thread: dict) -> None:
    try:
        tid = thread["thread_id"]

        title = thread.get(
            "title",
            "Untitled"
        )

        model_keys = llms.list_model_keys()
        model_key = thread.get(
            "model",
            model_keys[0] if model_keys else "llama-8b-instant"
        )

        icon = llms.get_icon(model_key)

        is_active = (
            st.session_state["thread_id"]
            == tid
        )

        # editing state
        if st.session_state.get("editing_thread") == tid:

            c1, c2, c3 = st.sidebar.columns([5, 1, 1])

            with c1:
                new = st.text_input(
                    "Edit",
                    value=title,
                    key=f"in_{tid}",
                    label_visibility="collapsed"
                )

            with c2:
                if st.button("✓", key=f"s_{tid}"):
                    _save_title(tid, new)

            with c3:
                if st.button("✗", key=f"c_{tid}"):

                    st.session_state["editing_thread"] = None

                    st.rerun()

            return

        label = f"{'📝 ' if is_active else ''}{icon} {title}"

        with st.sidebar.container():

            c1, c2, c3 = st.columns([6, 1, 1])

            with c1:
                if st.button(
                    label,
                    key=f"t_{tid}",
                    use_container_width=True
                ):
                    _switch_thread(tid)

            with c2:
                if st.button("✏️", key=f"e_{tid}"):

                    st.session_state["editing_thread"] = tid

                    st.rerun()

            with c3:
                if st.button("🗑️", key=f"d_{tid}"):

                    st.session_state["delete_confirm"] = tid

                    st.rerun()
    except Exception as e:
        st.sidebar.error(f"Failed to render thread: {str(e)}")


def _render_sidebar() -> None:
    try:
        _render_branding()
        _render_model_selector()

        if st.sidebar.button(
            "➕ New Chat",
            use_container_width=True,
            type="primary"
        ):
            _new_chat()

        st.sidebar.header("My Conversations")

        threads = list_threads()
        for thread in threads:
            _render_thread_row(thread)
    except Exception as e:
        st.sidebar.error(f"Failed to render sidebar: {str(e)}")


# ══════════════════════════════════════════════════════════════════════════════
# Chat UI
# ══════════════════════════════════════════════════════════════════════════════

def _render_header() -> None:
    try:
        tid = st.session_state["thread_id"]

        title = get_thread_title(tid)

        model_key = get_thread_model(tid)

        st.title(f"💬 {title}")

        st.caption(
            f"Using: {llms.get_icon(model_key)} "
            f"{llms.get_display_name(model_key)}"
        )
    except Exception as e:
        st.error(f"Failed to render header: {str(e)}")


def _render_history() -> None:
    try:
        for msg in st.session_state.get(
            "message_history",
            []
        ):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
    except Exception as e:
        st.error(f"Failed to render history: {str(e)}")


def _handle_input(user_input: str) -> None:
    if (
        not user_input
        or st.session_state.get(
            "processing_message"
        )
    ):
        return

    tid = st.session_state["thread_id"]

    model_key = get_thread_model(tid)

    # show user message instantly
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state["message_history"].append(
        {
            "role": "user",
            "content": user_input
        }
    )

    st.session_state["processing_message"] = True

    ai_response = ""

    try:
        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("▌")

            config = {
                "configurable": {
                    "thread_id": tid
                }
            }

            inputs = {
                "messages": [
                    HumanMessage(
                        content=user_input
                    )
                ],
                "model": model_key,
            }

            # Stream the response with better error handling for MySQL packet errors
            try:
                for chunk in chatbot.stream(
                    inputs,
                    config=config,
                    stream_mode="values"
                ):
                    msgs = chunk.get("messages", [])
                    if msgs:
                        last = msgs[-1]
                        if (
                            isinstance(last, AIMessage)
                            and last.content
                        ):
                            ai_response = last.content
                            placeholder.markdown(
                                ai_response + "▌"
                            )
            except Exception as stream_error:
                # Handle MySQL packet sequence error specifically
                error_msg = str(stream_error)
                if "Packet sequence number wrong" in error_msg:
                    print(
                        "[main._handle_input] MySQL packet sequence error - reconnecting...")
                    ai_response = "⚠️ Database connection issue. Please refresh the page and try again."
                    placeholder.markdown(ai_response)
                elif "BrokenPipeError" in error_msg or "ConnectionError" in error_msg:
                    print(
                        f"[main._handle_input] Connection error: {stream_error}")
                    ai_response = "⚠️ Connection lost. Please refresh the page and try again."
                    placeholder.markdown(ai_response)
                else:
                    print(f"[main._handle_input] Stream error: {stream_error}")
                    print(traceback.format_exc())
                    ai_response = f"⚠️ Error: {str(stream_error)}"
                    placeholder.markdown(ai_response)

            if not ai_response:
                ai_response = "⚠️ No response received. Please try again."
                placeholder.markdown(ai_response)

        # save response if it's a valid response (not an error message)
        if ai_response and not ai_response.startswith("⚠️"):
            st.session_state["message_history"].append(
                {
                    "role": "assistant",
                    "content": ai_response
                }
            )

            # auto title first exchange
            if len(
                st.session_state[
                    "message_history"
                ]
            ) == 2:
                try:
                    new_title = generate_title(
                        user_input,
                        model_key
                    )
                    update_title(
                        tid,
                        new_title
                    )
                except Exception as title_error:
                    print(
                        f"[main._handle_input] Title generation error: {title_error}")

    except Exception as exc:
        st.error(f"⚠️ Error: {exc}")
        print(f"[main._handle_input] Error: {exc}")
        print(traceback.format_exc())

        # Remove the user message if AI failed
        if st.session_state["message_history"]:
            st.session_state["message_history"].pop()

    finally:
        st.session_state["processing_message"] = False
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    try:
        _bootstrap()
        _render_sidebar()
        _render_header()

        # Always render history (welcome message removed)
        _render_history()

        user_input = st.chat_input(
            "Type your message here…"
        )

        if user_input:
            _handle_input(user_input)
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.stop()


if __name__ == "__main__":
    main()
