/**
 * static/js/utils.js
 * ───────────────────
 * Lightweight browser-side helpers injected once per page load.
 * All functions are side-effect-free at module level — safe to re-inject
 * on Streamlit reruns.
 */

/**
 * Scroll the chat area smoothly to the bottom after a new message arrives.
 */
function scrollChatToBottom() {
    const targets = document.querySelectorAll(
        '[data-testid="stChatMessageContainer"], .main'
    );
    targets.forEach(el => el.scrollTo({ top: el.scrollHeight, behavior: "smooth" }));
}

/**
 * Auto-resize a <textarea> to fit its content up to maxRows lines.
 * @param {HTMLTextAreaElement} ta
 * @param {number} maxRows
 */
function autoResizeTextarea(ta, maxRows = 6) {
    ta.style.height = "auto";
    const lh = parseInt(getComputedStyle(ta).lineHeight, 10) || 20;
    ta.style.height = Math.min(ta.scrollHeight, lh * maxRows) + "px";
}

/**
 * Watch the DOM for the Streamlit chat input and attach auto-resize once found.
 */
function attachChatInputResize() {
    const obs = new MutationObserver(() => {
        const ta = document.querySelector('[data-testid="stChatInput"] textarea');
        if (ta && !ta.dataset.resizeAttached) {
            ta.dataset.resizeAttached = "true";
            ta.addEventListener("input", () => autoResizeTextarea(ta));
        }
    });
    obs.observe(document.body, { childList: true, subtree: true });
}

// ── Boot ──────────────────────────────────────────────────────────────────────
(function init() {
    attachChatInputResize();
})();
