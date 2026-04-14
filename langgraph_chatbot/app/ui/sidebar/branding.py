"""Branding header and theme toggle component."""

import streamlit as st


def render_branding() -> None:
    """Render the branding header with a Python-driven theme toggle."""
    try:
        # ── 1. Initialise theme state ─────────────────────────────────────────
        if "theme" not in st.session_state:
            st.session_state["theme"] = "dark"

        is_light = st.session_state["theme"] == "light"

        # ── 2. Inject CSS variables inline ───────────────────────────────────
        if is_light:
            theme_css = _get_light_theme_css()
        else:
            theme_css = _get_dark_theme_css()

        st.markdown(theme_css, unsafe_allow_html=True)

        # ── 3. Branding HTML ─────────────────────────────────────────────────
        pill_class = "theme-switch is-light" if is_light else "theme-switch"
        st.sidebar.markdown(f"""
        <div class="branding-bar">
            <div class="branding-left">
                <span style="font-size:1.6rem;">🧠</span>
                <div>
                    <div class="branding-title">LangGraph Chat</div>
                    <div class="branding-subtitle">COM726 · DISSERTATION</div>
                </div>
            </div>
            <div class="{pill_class}" aria-hidden="true">
                <span class="switch-moon">🌙</span>
                <span class="switch-thumb"></span>
                <span class="switch-sun">☀️</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── 4. Actual toggle ─────────────────────────────────────────────────
        toggle_label = "Switch to 🌙 Dark" if is_light else "Switch to ☀️ Light"
        if st.sidebar.button(toggle_label, key="theme_toggle_btn",
                             use_container_width=True):
            st.session_state["theme"] = "dark" if is_light else "light"
            st.rerun()

    except Exception as e:
        st.sidebar.error(f"Failed to load branding: {str(e)}")


def _get_light_theme_css() -> str:
    """Return light theme CSS."""
    return """
    <style>
    :root {
        --accent:              #6C63FF;
        --accent-light:        #5a52e0;
        --border:              rgba(108, 99, 255, 0.22);
        --border-hover:        rgba(108, 99, 255, 0.50);
        --text-primary:        #1A1A2E;
        --text-muted:          #6b6897;
        --thread-bg:           rgba(108, 99, 255, 0.06);
        --thread-border:       rgba(108, 99, 255, 0.20);
        --thread-hover-bg:     rgba(108, 99, 255, 0.14);
        --thread-hover-border: rgba(108, 99, 255, 0.40);
        --text-thread:         #3a3660;
    }
    
    .stApp                                      { background-color: #F5F5FF !important; color: #1A1A2E !important; }
    section[data-testid="stSidebar"]            { background-color: #eeeeff !important; }
    header[data-testid="stHeader"],
    [data-testid="stToolbar"],
    .stAppToolbar                               { background-color: #eeeeff !important; border-bottom: 1px solid rgba(108,99,255,0.15) !important; }
    header[data-testid="stHeader"] *,
    [data-testid="stToolbar"] *                 { color: #1A1A2E !important; }
    .stApp p, .stApp span, .stApp div,
    .stApp h1, .stApp h2, .stApp h3,
    .stApp h4, .stApp h5, .stApp h6,
    .stApp label, .stApp li, .stApp a,
    [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] *,
    [data-testid="stChatMessage"] *,
    [data-testid="stText"]                      { color: #1A1A2E !important; }
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] .stSubheader { color: #6C63FF !important; }
    section[data-testid="stSidebar"] .stCaption,
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div:not(.branding-bar) { color: #1A1A2E !important; }
    [data-baseweb="select"],
    [data-baseweb="select"] > div,
    [data-baseweb="select"] input               { background-color: #ffffff !important; color: #1A1A2E !important; border-color: rgba(108,99,255,0.3) !important; }
    [data-baseweb="popover"] *,
    [data-baseweb="menu"] *                     { background-color: #ffffff !important; color: #1A1A2E !important; }
    [data-testid="stFileUploadDropzone"] {
        background-color: #ffffff !important;
        border: 2px dashed rgba(108, 99, 255, 0.3) !important;
        border-radius: 8px !important;
        padding: 20px !important;
    }
    [data-testid="stFileUploadDropzone"] > div { background-color: #ffffff !important; }
    [data-testid="stFileUploadDropzone"] * { background-color: transparent !important; color: #1A1A2E !important; }
    [data-testid="stFileUploadDropzone"] svg { fill: #6C63FF !important; stroke: #6C63FF !important; }
    [data-testid="stFileUploadDropzone"] p,
    [data-testid="stFileUploadDropzone"] span { color: #6b6897 !important; }
    [data-testid="stFileUploadDropzone"] button { background-color: #e8e7ff !important; color: #6C63FF !important; border: 1px solid rgba(108, 99, 255, 0.3) !important; border-radius: 6px !important; }
    [data-testid="stFileUploadDropzone"] button:hover { background-color: #d8d6ff !important; }
    [data-testid="stBottom"], [data-testid="stBottom"] > div { background-color: #F5F5FF !important; }
    [data-testid="stChatInput"], [data-testid="stChatInput"] > div { background-color: #ffffff !important; border: 1px solid rgba(108,99,255,0.3) !important; border-radius: 12px !important; }
    [data-testid="stChatInput"] textarea { background-color: #ffffff !important; color: #1A1A2E !important; }
    [data-testid="stChatInput"] button { color: #6C63FF !important; }
    section[data-testid="stSidebar"] .stButton > button { background-color: #e8e7ff !important; color: #1A1A2E !important; border: 1px solid rgba(108, 99, 255, 0.25) !important; }
    section[data-testid="stSidebar"] .stButton > button:hover { background-color: #d8d6ff !important; }
    section[data-testid="stSidebar"] .stButton > button[kind="primary"] { background-color: #6C63FF !important; color: white !important; }
    </style>
    """


def _get_dark_theme_css() -> str:
    """Return dark theme CSS."""
    return """
    <style>
    :root {
        --accent:              #6C63FF;
        --accent-light:        #9f9aff;
        --border:              rgba(108, 99, 255, 0.15);
        --border-hover:        rgba(108, 99, 255, 0.40);
        --text-primary:        #e0e0f0;
        --text-muted:          #7a75a8;
        --thread-bg:           rgba(255, 255, 255, 0.04);
        --thread-border:       rgba(108, 99, 255, 0.20);
        --thread-hover-bg:     rgba(108, 99, 255, 0.12);
        --thread-hover-border: rgba(108, 99, 255, 0.45);
        --text-thread:         #b8b4e8;
    }
    
    [data-testid="stFileUploadDropzone"] {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border: 2px dashed rgba(108, 99, 255, 0.3) !important;
        border-radius: 8px !important;
    }
    [data-testid="stFileUploadDropzone"] * { color: #e0e0f0 !important; }
    [data-testid="stFileUploadDropzone"] button { background-color: rgba(108, 99, 255, 0.2) !important; color: #c4c0ff !important; }
    </style>
    """
