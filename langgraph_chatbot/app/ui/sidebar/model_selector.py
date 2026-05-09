"""Model selector component."""

import streamlit as st
import llms
from app.ui.sidebar.thread_actions import on_model_change


def _inject_css() -> None:
    """Inject CSS for model selector styling."""
    if "model_selector_css_injected" not in st.session_state:
        st.markdown("""
            <style>
            /* Target the dropdown menu container */
            [data-testid="stSelectbox"] [data-baseweb="select"] [data-testid="stMarkdownContainer"] {
                max-height: 600px !important;
                overflow-y: auto !important;
            }
            
            /* Target the virtualized list container */
            [data-baseweb="select"] [data-simplebar] {
                max-height: 600px !important;
            }
            
            /* Target all dropdown menu containers */
            .stSelectbox .select__menu {
                max-height: 600px !important;
            }
            
            .stSelectbox .select__menu-list {
                max-height: 600px !important;
            }
            
            /* Make dropdown items taller */
            .stSelectbox .select__option {
                padding: 12px 16px !important;
                min-height: 50px !important;
                white-space: normal !important;
            }
            
            /* Override any inline max-height styles */
            div[role="listbox"] {
                max-height: 600px !important;
                height: auto !important;
            }
            
            /* Ensure the popover doesn't clip content */
            [data-baseweb="popover"] {
                overflow: visible !important;
            }
            
            [data-baseweb="popover"] [data-testid="stSelectbox"] {
                overflow: visible !important;
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
