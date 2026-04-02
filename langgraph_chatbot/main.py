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
from app.utils.thread_service import init_threads_table

_ROOT = Path(__file__).resolve().parent.parent  # points to COM726/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ── SQL table queries ─────────────────────────────────────────────
init_threads_table()

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
        # Clear any pending edit/delete states
        st.session_state["editing_thread"] = None
        st.session_state["delete_confirm"] = None

        st.session_state.update({
            "thread_id": thread_id,
            "message_history": load_conversation(thread_id),
            "selected_model": get_thread_model(thread_id),
        })

        st.rerun()
    except Exception as e:
        st.error(f"Failed to switch thread: {str(e)}")


def _delete_thread(thread_id: str) -> None:
    try:
        # First, delete from thread service (MySQL)
        success = delete_thread(thread_id)

        if success:
            # Also delete from checkpoint store if possible
            try:
                if hasattr(chatbot, 'checkpointer') and hasattr(chatbot.checkpointer, 'delete_thread'):
                    chatbot.checkpointer.delete_thread(thread_id)
            except Exception as e:
                print(
                    f"Warning: Could not delete checkpoints for {thread_id}: {e}")

            # If the deleted thread was the current one, switch to another thread
            if st.session_state.get("thread_id") == thread_id:
                recent = get_most_recent_thread_id()

                if recent and recent != thread_id:
                    # Switch to most recent thread
                    st.session_state.update({
                        "thread_id": recent,
                        "message_history": load_conversation(recent),
                        "selected_model": get_thread_model(recent),
                    })
                else:
                    # No threads left, create a new one
                    _new_chat()
                    return

        # Clear confirmation state
        st.session_state["delete_confirm"] = None
        st.rerun()

    except Exception as e:
        st.error(f"Failed to delete thread: {str(e)}")
        print(f"Delete thread error: {traceback.format_exc()}")


def _save_title(thread_id: str, new_title: str) -> None:
    try:
        if new_title and new_title.strip():
            success = update_title(thread_id, new_title.strip())
            if success:
                st.toast(f"Title updated to: {new_title.strip()}", icon="✅")
            else:
                st.warning("Failed to update title")

        # Clear editing state
        st.session_state["editing_thread"] = None
        st.rerun()
    except Exception as e:
        st.error(f"Failed to save title: {str(e)}")
        print(f"Save title error: {traceback.format_exc()}")


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
# Sidebar components
# ══════════════════════════════════════════════════════════════════════════════

def _render_branding() -> None:
    """Render the branding/logo section in sidebar."""
    try:
        # Try to load HTML template if exists
        html = load_template("sidebar.html")
        if html:
            st.sidebar.markdown(html, unsafe_allow_html=True)
        else:
            # Fallback branding
            st.sidebar.markdown("""
            <div style="text-align: center; padding: 1rem 0;">
                <h2 style="margin: 0; color: #FF4B4B;">🧠 COM726</h2>
                <p style="margin: 0; font-size: 0.8rem; opacity: 0.8;">LangGraph Chatbot</p>
            </div>
            """, unsafe_allow_html=True)
            st.sidebar.divider()
    except Exception as e:
        st.sidebar.error(f"Failed to load branding: {str(e)}")


def _render_model_selector() -> None:
    """Render the model selector dropdown in sidebar."""
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
            keys[0] if keys else None
        )

        # Find index of current model
        idx = keys.index(current) if current in keys else 0

        st.sidebar.subheader("🤖 Model")

        st.sidebar.selectbox(
            "Choose AI Model",
            options=keys,
            format_func=lambda k: display_map.get(k, k),
            index=idx,
            key="model_selector",
            on_change=_on_model_change,
            help="Select which AI model to use for this conversation"
        )

        st.sidebar.caption("⚡ Powered by Groq's free tier")
        st.sidebar.divider()
    except Exception as e:
        st.sidebar.error(f"Failed to load model selector: {str(e)}")


def _render_delete_confirmation() -> None:
    """Render a confirmation dialog for thread deletion."""
    thread_id = st.session_state.get("delete_confirm")

    if not thread_id:
        return

    # Get thread title for display
    thread_title = get_thread_title(thread_id)

    # Create a modal-like confirmation
    with st.sidebar:
        st.markdown("### ⚠️ Delete Conversation")
        st.markdown(f"Are you sure you want to delete **{thread_title}**?")
        st.markdown("This action cannot be undone.")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Yes, Delete", key="confirm_delete", use_container_width=True):
                _delete_thread(thread_id)
        with col2:
            if st.button("Cancel", key="cancel_delete", use_container_width=True):
                st.session_state["delete_confirm"] = None
                st.rerun()

        st.divider()


def _render_edit_title(thread: dict) -> None:
    """Render inline edit title interface."""
    tid = thread["thread_id"]
    current_title = thread.get("title", "Untitled")

    with st.sidebar:
        # Create a compact edit form
        st.markdown("---")
        st.markdown("**✏️ Edit Title**")

        new_title = st.text_input(
            "Title",
            value=current_title,
            key=f"edit_input_{tid}",
            label_visibility="collapsed"
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save", key=f"save_{tid}", use_container_width=True):
                _save_title(tid, new_title)
        with col2:
            if st.button("Cancel", key=f"cancel_{tid}", use_container_width=True):
                st.session_state["editing_thread"] = None
                st.rerun()


def _render_thread_row(thread: dict) -> None:
    """Render a single thread row with edit/delete buttons."""
    try:
        tid = thread["thread_id"]
        title = thread.get("title", "Untitled")

        model_keys = llms.list_model_keys()
        model_key = thread.get(
            "model", model_keys[0] if model_keys else "llama-8b-instant")

        icon = llms.get_icon(model_key)
        is_active = st.session_state.get("thread_id") == tid

        # Check if this thread is currently being edited (global edit mode)
        if st.session_state.get("editing_thread") == tid:
            _render_edit_title(thread)
            return

        # Prepare label with active indicator
        label = f"{'📌 ' if is_active else ''}{icon} {title[:40]}{'...' if len(title) > 40 else ''}"

        # Create three columns: thread button, edit, delete
        col1, col2, col3 = st.sidebar.columns([6, 1, 1])

        with col1:
            # Thread button (with custom styling for active thread)
            button_type = "primary" if is_active else "secondary"
            if st.button(
                label,
                key=f"t_{tid}",
                use_container_width=True,
                type=button_type
            ):
                _switch_thread(tid)

        with col2:
            # Edit button - opens edit mode for this thread
            if st.button("✏️", key=f"e_{tid}", help="Edit title"):
                st.session_state["editing_thread"] = tid
                # Clear any delete confirmation
                st.session_state["delete_confirm"] = None
                st.rerun()

        with col3:
            # Delete button - shows confirmation dialog
            if st.button("🗑️", key=f"d_{tid}", help="Delete conversation"):
                st.session_state["delete_confirm"] = tid
                # Clear any edit mode
                st.session_state["editing_thread"] = None
                st.rerun()

    except Exception as e:
        st.sidebar.error(f"Failed to render thread: {str(e)}")
        print(f"Render thread error: {traceback.format_exc()}")


def _render_sidebar() -> None:
    """Render the complete sidebar with all threads."""
    try:
        _render_branding()
        _render_model_selector()

        # New Chat button
        if st.sidebar.button(
            "➕ New Chat",
            use_container_width=True,
            type="primary"
        ):
            _new_chat()

        st.sidebar.divider()
        st.sidebar.header("📝 Conversations")

        # Get all threads from database
        threads = list_threads()

        if not threads:
            st.sidebar.info("No conversations yet. Start a new chat!")
        else:
            # Render all threads
            for thread in threads:
                _render_thread_row(thread)

        # Render delete confirmation if needed
        _render_delete_confirmation()

    except Exception as e:
        st.sidebar.error(f"Failed to render sidebar: {str(e)}")
        print(f"Render sidebar error: {traceback.format_exc()}")


# ══════════════════════════════════════════════════════════════════════════════
# Chat UI
# ══════════════════════════════════════════════════════════════════════════════

def _render_header() -> None:
    """Render the main chat header."""
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


def _render_history() -> None:
    """Render the conversation history."""
    try:
        for msg in st.session_state.get("message_history", []):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
    except Exception as e:
        st.error(f"Failed to render history: {str(e)}")


def _handle_input(user_input: str) -> None:
    """Handle user input and generate AI response."""
    if not user_input or st.session_state.get("processing_message"):
        return

    tid = st.session_state["thread_id"]
    model_key = get_thread_model(tid)

    # Show user message instantly
    with st.chat_message("user"):
        st.markdown(user_input)

    st.session_state["message_history"].append({
        "role": "user",
        "content": user_input
    })

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
                "messages": [HumanMessage(content=user_input)],
                "model": model_key,
            }

            # Stream the response
            try:
                for chunk in chatbot.stream(
                    inputs,
                    config=config,
                    stream_mode="values"
                ):
                    msgs = chunk.get("messages", [])
                    if msgs:
                        last = msgs[-1]
                        if isinstance(last, AIMessage) and last.content:
                            ai_response = last.content
                            placeholder.markdown(ai_response + "▌")
            except Exception as stream_error:
                error_msg = str(stream_error)
                if "Packet sequence number wrong" in error_msg:
                    ai_response = "⚠️ Database connection issue. Please refresh the page and try again."
                    placeholder.markdown(ai_response)
                elif "BrokenPipeError" in error_msg or "ConnectionError" in error_msg:
                    ai_response = "⚠️ Connection lost. Please refresh the page and try again."
                    placeholder.markdown(ai_response)
                else:
                    print(f"Stream error: {stream_error}")
                    print(traceback.format_exc())
                    ai_response = f"⚠️ Error: {str(stream_error)}"
                    placeholder.markdown(ai_response)

            if not ai_response:
                ai_response = "⚠️ No response received. Please try again."
                placeholder.markdown(ai_response)

        # Save response if it's valid
        if ai_response and not ai_response.startswith("⚠️"):
            st.session_state["message_history"].append({
                "role": "assistant",
                "content": ai_response
            })

            # Auto-generate title for first exchange
            if len(st.session_state["message_history"]) == 2:
                try:
                    new_title = generate_title(user_input, model_key)
                    update_title(tid, new_title)
                    st.toast(f"Chat titled: {new_title}", icon="✨")
                except Exception as title_error:
                    print(f"Title generation error: {title_error}")

    except Exception as exc:
        st.error(f"⚠️ Error: {exc}")
        print(f"Error in _handle_input: {exc}")
        print(traceback.format_exc())

        # Remove the user message if AI failed
        if st.session_state["message_history"]:
            st.session_state["message_history"].pop()

    finally:
        st.session_state["processing_message"] = False
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Main application entry point."""
    try:
        _bootstrap()
        _render_sidebar()
        _render_header()
        _render_history()

        user_input = st.chat_input("Type your message here…")
        if user_input:
            _handle_input(user_input)

    except Exception as e:
        st.error(f"Application error: {str(e)}")
        print(f"Main error: {traceback.format_exc()}")
        st.stop()


if __name__ == "__main__":
    main()
