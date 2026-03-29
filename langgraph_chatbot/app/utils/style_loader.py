"""
app/utils/style_loader.py
──────────────────────────
Reads static CSS / JS files from disk and injects them into the Streamlit
page via st.markdown.  Caching with @st.cache_data means each file is read
from disk only once per Streamlit server process.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from config.settings import STATIC_DIR


@st.cache_data(show_spinner=False)
def _read_file(path: str) -> str:
    """Read a text file from disk (cached)."""
    return Path(path).read_text(encoding="utf-8")


def inject_css(*css_filenames: str) -> None:
    """
    Inject one or more CSS files from ``static/css/`` into the page.

    Parameters
    ----------
    *css_filenames:
        File names relative to ``static/css/``, e.g. ``"chat.css"``.
    """
    for filename in css_filenames:
        css_path = str(STATIC_DIR / "css" / filename)
        try:
            css = _read_file(css_path)
            st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
        except FileNotFoundError:
            st.warning(f"[style_loader] CSS file not found: {css_path}")


def inject_js(*js_filenames: str) -> None:
    """
    Inject one or more JS files from ``static/js/`` into the page.

    Parameters
    ----------
    *js_filenames:
        File names relative to ``static/js/``, e.g. ``"utils.js"``.
    """
    for filename in js_filenames:
        js_path = str(STATIC_DIR / "js" / filename)
        try:
            js = _read_file(js_path)
            st.markdown(f"<script>{js}</script>", unsafe_allow_html=True)
        except FileNotFoundError:
            st.warning(f"[style_loader] JS file not found: {js_path}")


def load_html_template(template_filename: str) -> str:
    """
    Read an HTML template from ``templates/`` and return its content as a
    string so the caller can interpolate values before injecting it.

    Parameters
    ----------
    template_filename:
        File name relative to ``templates/``, e.g. ``"chat.html"``.
    """
    from config.settings import TEMPLATES_DIR

    template_path = str(TEMPLATES_DIR / template_filename)
    try:
        return _read_file(template_path)
    except FileNotFoundError:
        st.warning(f"[style_loader] Template not found: {template_path}")
        return ""


load_template = load_html_template
