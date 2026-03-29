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
from langchain_core.messages import HumanMessage
import streamlit as st
import llms
import sys
from pathlib import Path

# ── FIX PATH FIRST: Ensure COM726/ root is on sys.path ──────────────────────
# This must happen before 'import llms' or 'from app.utils import ...'
_ROOT = Path(__file__).resolve().parent.parent   # Points to COM726/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Standard & Local Imports ──────────────────────────────────────────────────

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
        st.session_state["thread_id"] = tid
        st.session_state["message_history"] = load_conversation(tid)
        default_model = first.get("model", default_model)

    st.session_state["selected_model"] = default_model
    st.session_state["editing_thread"] = None
    st.session_state["delete_confirm"] = None
    st.session_state["processing_message"] = False
    st.session_state["_initialized"] = True

# ══════════════════════════════════════════════════════════════════════════════
# Thread action handlers
# ══════════════════════════════════════════════════════════════════════════════


def _new_chat() -> None:
    tid = new_thread_id()
    model = st.session_state.get("selected_model", llms.list_model_keys()[0])
    create_thread(tid, title="New Chat", model=model)
    st.session_state.update({
        "thread_id":       tid,
        "message_history": [],
        "editing_thread":  None,
        "delete_confirm":  None,
    })
    st.rerun()


def _switch_thread(thread_id: str) -> None:
    if st.session_state["thread_id"] == thread_id:
        return
    st.session_state.update({
        "thread_id":       thread_id,
        "message_history": load_conversation(thread_id),
        "selected_model":  get_thread_model(thread_id),
        "editing_thread":  None,
        "delete_confirm":  None,
    })
    st.rerun()


def _delete_thread(thread_id: str) -> None:
    delete_thread(thread_id)
    if st.session_state["thread_id"] == thread_id:
        recent = get_most_recent_thread_id()
        if recent:
            st.session_state.update({
                "thread_id":       recent,
                "message_history": load_conversation(recent),
                "selected_model":  get_thread_model(recent),
            })
        else:
            _new_chat()
            return
    st.session_state["delete_confirm"] = None
    st.rerun()


def _save_title(thread_id: str, new_title: str) -> None:
    if new_title.strip():
        update_title(thread_id, new_title.strip())
    st.session_state["editing_thread"] = None
    st.rerun()


def _on_model_change() -> None:
    chosen = st.session_state["model_selector"]
    st.session_state["selected_model"] = chosen
    tid = st.session_state.get("thread_id")
    if tid:
        update_model(tid, chosen)

# ══════════════════════════════════════════════════════════════════════════════
# Sidebar rendering
# ══════════════════════════════════════════════════════════════════════════════


def _render_branding() -> None:
    html = load_template("sidebar.html")
    if html:
        st.sidebar.markdown(html, unsafe_allow_html=True)


def _render_model_selector() -> None:
    keys = llms.list_model_keys()
    display_map = {k: llms.get_display_name(k) for k in keys}
    current = st.session_state.get("selected_model", keys[0])
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


def _render_thread_row(thread: dict) -> None:
    tid = thread["thread_id"]
    title = thread.get("title", "Untitled")
    model_key = thread.get("model", llms.list_model_keys()[0])
    icon = llms.get_icon(model_key)
    is_active = st.session_state["thread_id"] == tid

    if st.session_state.get("editing_thread") == tid:
        c1, c2, c3 = st.sidebar.columns([5, 1, 1])
        with c1:
            new = st.text_input("Edit", value=title,
                                key=f"in_{tid}", label_visibility="collapsed")
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
            if st.button(label, key=f"t_{tid}", use_container_width=True):
                _switch_thread(tid)
        with c2:
            if st.button("✏️", key=f"e_{tid}"):
                st.session_state["editing_thread"] = tid
                st.rerun()
        with c3:
            if st.button("🗑️", key=f"d_{tid}"):
                st.session_state["delete_confirm"] = tid
                st.rerun()


def _render_sidebar() -> None:
    _render_branding()
    _render_model_selector()
    if st.sidebar.button("➕ New Chat", use_container_width=True, type="primary"):
        _new_chat()
    st.sidebar.header("My Conversations")
    for thread in list_threads():
        _render_thread_row(thread)

# ══════════════════════════════════════════════════════════════════════════════
# Main chat area rendering
# ══════════════════════════════════════════════════════════════════════════════


def _render_header() -> None:
    tid = st.session_state["thread_id"]
    title = get_thread_title(tid)
    model_key = get_thread_model(tid)
    st.title(f"💬 {title}")
    st.caption(
        f"Using: {llms.get_icon(model_key)} {llms.get_display_name(model_key)}")


def _render_history() -> None:
    for msg in st.session_state.get("message_history", []):
        with st.chat_message(msg["role"]):
            st.write(msg["content"])


def _handle_input(user_input: str) -> None:
    if not user_input or st.session_state.get("processing_message"):
        return
    st.session_state["processing_message"] = True
    st.session_state["message_history"].append(
        {"role": "user", "content": user_input})

    tid = st.session_state["thread_id"]
    try:
        # Simplified streaming call for brevity
        with st.chat_message("assistant"):
            placeholder = st.empty()
            ai_response = ""
            for chunk in chatbot.stream({"messages": [HumanMessage(content=user_input)], "model": get_thread_model(tid)}, config={"configurable": {"thread_id": tid}}, stream_mode="values"):
                if "messages" in chunk and chunk["messages"]:
                    ai_response = chunk["messages"][-1].content
                    placeholder.write(ai_response)

            st.session_state["message_history"].append(
                {"role": "assistant", "content": ai_response})
            if len(st.session_state["message_history"]) <= 2:
                update_title(tid, generate_title(
                    user_input, get_thread_model(tid)))
    except Exception as e:
        st.error(f"Error: {e}")
    finally:
        st.session_state["processing_message"] = False
        st.rerun()


def main() -> None:
    _bootstrap()
    _render_sidebar()
    _render_header()
    _render_history()
    user_input = st.chat_input("Type here...")
    if user_input:
        _handle_input(user_input)


if __name__ == "__main__":
    main()
