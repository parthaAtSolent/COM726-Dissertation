"""
app/utils/file_uploader.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Custom file-uploader Streamlit component.

Expected project layout (all paths relative to langgraph_chatbot/):
  static/css/file_uploader.css
  static/js/file_uploader.js
  templates/file_uploader.html
  _component_build/          ← auto-created, add to .gitignore

At import time this module:
  1. Reads  templates/file_uploader.html
  2. Reads  static/css/file_uploader.css
  3. Reads  static/js/file_uploader.js
  4. Replaces <!--__STYLE_BLOCK__--> with a full <style>…</style> block
     and <!--__SCRIPT_BLOCK__--> with a full <script>…</script> block
  5. Writes the assembled index.html to _component_build/file_uploader/
  6. Registers a Streamlit declared component pointing at that directory
"""

from __future__ import annotations

import pathlib

import streamlit.components.v1 as components

# ── Resolve paths ─────────────────────────────────────────────────────────────
#
#   __file__        →  langgraph_chatbot/app/utils/file_uploader.py
#   .parent         →  langgraph_chatbot/app/utils/
#   .parent         →  langgraph_chatbot/app/
#   .parent         →  langgraph_chatbot/             ← _ROOT
#
_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent

_TEMPLATE_FILE = _ROOT / "templates" / "file_uploader.html"
_CSS_FILE = _ROOT / "static" / "css" / "file_uploader.css"
_JS_FILE = _ROOT / "static" / "js" / "file_uploader.js"
_BUILD_DIR = _ROOT / "_component_build" / "file_uploader"
_BUILD_FILE = _BUILD_DIR / "index.html"

_STYLE_MARKER = "<!--__STYLE_BLOCK__-->"
_SCRIPT_MARKER = "<!--__SCRIPT_BLOCK__-->"


# ── Startup check — fail early with a helpful message ─────────────────────────

def _check_sources() -> None:
    missing = [p for p in (_TEMPLATE_FILE, _CSS_FILE,
                           _JS_FILE) if not p.exists()]
    if missing:
        lines = "\n".join(f"  ✗  {p}" for p in missing)
        raise FileNotFoundError(
            "\n\n[file_uploader] Missing source files:\n"
            f"{lines}\n\n"
            "Expected layout inside langgraph_chatbot/:\n"
            "  templates/file_uploader.html\n"
            "  static/css/file_uploader.css\n"
            "  static/js/file_uploader.js\n"
        )


# ── Assemble index.html ────────────────────────────────────────────────────────

def _build_component() -> pathlib.Path:
    """Inline CSS + JS into the HTML template and write to the build directory."""
    _check_sources()

    sources = [_TEMPLATE_FILE, _CSS_FILE, _JS_FILE]

    # Skip rebuild when nothing has changed
    if _BUILD_FILE.exists():
        build_mtime = _BUILD_FILE.stat().st_mtime
        source_mtime = max(p.stat().st_mtime for p in sources)
        if source_mtime <= build_mtime:
            return _BUILD_DIR

    html = _TEMPLATE_FILE.read_text(encoding="utf-8")
    css = _CSS_FILE.read_text(encoding="utf-8")
    js = _JS_FILE.read_text(encoding="utf-8")

    html = html.replace(_STYLE_MARKER,  f"<style>\n{css}\n</style>")
    html = html.replace(_SCRIPT_MARKER, f"<script>\n{js}\n</script>")

    _BUILD_DIR.mkdir(parents=True, exist_ok=True)
    _BUILD_FILE.write_text(html, encoding="utf-8")

    print(f"[file_uploader] built → {_BUILD_FILE}")
    return _BUILD_DIR


# ── Register Streamlit component ──────────────────────────────────────────────

_build_dir = _build_component()
_component_func = components.declare_component(
    "custom_file_uploader",
    path=str(_build_dir),
)


# ── Public API ────────────────────────────────────────────────────────────────

def custom_file_uploader(
    accepted_types: list[str] | None = None,
    icon: str = "📁",
    label: str = "Drop files here or click to browse",
    max_files: int = 10,
    height: int = 160,
    key: str | None = None,
) -> list[dict] | None:
    """
    Render the custom file-upload widget.

    Returns
    -------
    None until the user selects files, then a list of dicts:
        [{ "name": str, "mime": str, "size": int, "b64": str }, ...]
    """
    accepted_types = accepted_types or ["pdf", "txt"]

    return _component_func(
        accepted_types=accepted_types,
        icon=icon,
        label=label,
        max_files=max_files,
        key=key,
        default=None,
        height=height,
    )
