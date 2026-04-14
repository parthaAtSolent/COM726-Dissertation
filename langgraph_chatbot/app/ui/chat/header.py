"""Chat header component."""

import streamlit as st
import llms
from app.utils import get_thread_title, get_thread_model
from app.rag import list_ingested_files


def render_header() -> None:
    """Render the chat header with title and model info."""
    try:
        tid = st.session_state.get("thread_id")
        if not tid:
            return

        title = get_thread_title(tid)
        model_key = get_thread_model(tid)

        ingested_files = list_ingested_files()
        rag_indicator = " 📚" if ingested_files else ""

        st.title(f"💬 {title}{rag_indicator}")

        if ingested_files:
            st.caption(
                f"Using: {llms.get_icon(model_key)} {llms.get_display_name(model_key)} "
                f"• 📄 {len(ingested_files)} document(s) loaded"
            )
        else:
            st.caption(
                f"Using: {llms.get_icon(model_key)} {llms.get_display_name(model_key)}"
            )
    except Exception as e:
        st.error(f"Failed to render header: {str(e)}")
