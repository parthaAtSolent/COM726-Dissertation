"""Model selector component."""

import streamlit as st
import llms
from app.ui.sidebar.thread_actions import on_model_change


def _inject_css() -> None:
    """Inject CSS for model selector styling. Only runs once per session."""
    if "model_selector_css_injected" not in st.session_state:
        st.markdown("""
            <style>
            /* Make the selectbox trigger button larger */
            div[data-testid="stSelectbox"] > div > div {
                font-size: 16px;
                min-height: 48px;
            }
            
            div[data-testid="stSelectbox"] > div > div > div {
                padding: 10px 14px;
            }
            
            /* Dropdown menu items styling */
            [data-baseweb="select"] ul {
                max-height: 500px !important;
                min-height: auto !important;
            }
            
            /* Individual dropdown items */
            [data-baseweb="select"] ul li {
                font-size: 15px;
                padding: 12px 14px !important;
                min-height: 48px;
                height: auto !important;
                display: flex;
                align-items: center;
                white-space: normal !important;
                word-wrap: break-word !important;
            }
            
            /* Popover/dropdown container */
            [data-baseweb="popover"] {
                max-height: 550px !important;
                overflow-y: auto !important;
            }
            
            [data-baseweb="popover"] ul {
                max-height: none !important;
            }
            
            /* Optional: Wider dropdown */
            div[data-testid="stSelectbox"] div[data-baseweb="select"] {
                width: 100%;
            }
            
            /* Allow text wrapping in dropdown items */
            .stSelectbox div[role="listbox"] div {
                white-space: normal !important;
                word-break: break-word !important;
            }
            </style>
        """, unsafe_allow_html=True)
        st.session_state.model_selector_css_injected = True


def render_model_selector() -> None:
    """Render the model selector dropdown with custom styling."""
    _inject_css()

    try:
        keys = llms.list_model_keys()
        if not keys:
            st.sidebar.error("No models available")
            return

        display_map = {k: llms.get_display_name(k) for k in keys}
        current = st.session_state.get(
            "selected_model", keys[0] if keys else None)
        idx = keys.index(current) if current in keys else 0

        st.sidebar.subheader("🤖 Model")

        st.sidebar.selectbox(
            "Choose AI Model",
            options=keys,
            format_func=lambda k: display_map.get(k, k),
            index=idx,
            key="model_selector_widget",
            on_change=on_model_change,
            help="Select which AI model to use for this conversation"
        )

        st.sidebar.caption("⚡ Powered by Groq's free tier")
        st.sidebar.divider()
    except Exception as e:
        st.sidebar.error(f"Failed to load model selector: {str(e)}")
