/**
 * static/js/theme.js
 * Complete working theme toggle for Streamlit
 * CopyCase-style animated sun/moon pill
 */

const THEME_KEY = "com726_theme";
const ACTIVE_CLASS = "is-light";

// Direct color application (more reliable than CSS vars in Streamlit)
function applyThemeToStreamlit(theme) {
    const isDark = theme === "dark";
    const root = document.documentElement;
    const sidebar = document.querySelector('section[data-testid="stSidebar"]');
    const mainApp = document.querySelector('.stApp');
    const chatInput = document.querySelector('[data-testid="stChatInput"] textarea');
    
    // 1. Update HTML class for CSS selectors
    root.classList.remove("theme-dark", "theme-light");
    root.classList.add(isDark ? "theme-dark" : "theme-light");
    
    // 2. Apply colors directly to sidebar (bypass Streamlit's CSS var issues)
    if (sidebar) {
        if (isDark) {
            sidebar.style.background = "linear-gradient(180deg, #1A1A2E 0%, #16213e 100%)";
            sidebar.style.borderRight = "1px solid rgba(108, 99, 255, 0.2)";
        } else {
            sidebar.style.background = "linear-gradient(180deg, #EBEBFF 0%, #E0E0F8 100%)";
            sidebar.style.borderRight = "1px solid rgba(108, 99, 255, 0.25)";
        }
    }
    
    // 3. Apply to main app area
    if (mainApp) {
        mainApp.style.backgroundColor = isDark ? "#0F0F1A" : "#F5F5FF";
        mainApp.style.color = isDark ? "#E0E0F0" : "#1A1A2E";
        mainApp.style.transition = "background-color 0.35s ease, color 0.35s ease";
    }
    
    // 4. Style chat input
    if (chatInput) {
        chatInput.style.backgroundColor = isDark ? "#1A1A2E" : "#EBEBFF";
        chatInput.style.color = isDark ? "#E0E0F0" : "#1A1A2E";
        chatInput.style.borderColor = isDark ? "rgba(108, 99, 255, 0.3)" : "rgba(108, 99, 255, 0.2)";
    }
    
    // 5. Update toggle pill appearance
    const toggle = document.getElementById("theme-toggle");
    if (toggle) {
        if (!isDark) {
            toggle.classList.add(ACTIVE_CLASS);
            toggle.setAttribute("aria-checked", "true");
        } else {
            toggle.classList.remove(ACTIVE_CLASS);
            toggle.setAttribute("aria-checked", "false");
        }
    }
    
    // 6. Update all dynamic elements
    updateDynamicElements(theme);
    updateChatMessages(theme);
}

// Update buttons, select boxes, and other dynamic elements
function updateDynamicElements(theme) {
    const isDark = theme === "dark";
    
    // Update sidebar buttons
    const buttons = document.querySelectorAll('section[data-testid="stSidebar"] .stButton button');
    buttons.forEach(btn => {
        if (btn.closest('[kind="primary"]') || btn.getAttribute('kind') === 'primary') {
            // Keep gradient for primary button
            btn.style.background = "linear-gradient(135deg, #6C63FF, #a78bfa)";
            btn.style.color = "#ffffff";
        } else {
            btn.style.backgroundColor = isDark ? "rgba(255, 255, 255, 0.04)" : "rgba(108, 99, 255, 0.06)";
            btn.style.color = isDark ? "#c8c4e8" : "#2e2b50";
            btn.style.border = `1px solid ${isDark ? "rgba(255, 255, 255, 0.08)" : "rgba(108, 99, 255, 0.15)"}`;
        }
    });
    
    // Update select boxes
    const selects = document.querySelectorAll('.stSelectbox > div[data-baseweb="select"]');
    selects.forEach(select => {
        select.style.backgroundColor = isDark ? "rgba(108, 99, 255, 0.1)" : "rgba(108, 99, 255, 0.08)";
        select.style.border = `1px solid ${isDark ? "rgba(108, 99, 255, 0.2)" : "rgba(108, 99, 255, 0.25)"}`;
        select.style.borderRadius = "8px";
    });
    
    // Update expanders and other elements
    const expanders = document.querySelectorAll('.streamlit-expanderHeader');
    expanders.forEach(expander => {
        expander.style.color = isDark ? "#E0E0F0" : "#1A1A2E";
        expander.style.backgroundColor = isDark ? "rgba(255, 255, 255, 0.03)" : "rgba(108, 99, 255, 0.04)";
    });
}

// Update chat messages
function updateChatMessages(theme) {
    const isDark = theme === "dark";
    const chatMessages = document.querySelectorAll('[data-testid="stChatMessage"]');
    chatMessages.forEach(msg => {
        msg.style.backgroundColor = isDark ? "rgba(255, 255, 255, 0.05)" : "rgba(108, 99, 255, 0.04)";
        msg.style.borderRadius = "12px";
        msg.style.padding = "12px";
        msg.style.margin = "8px 0";
        msg.style.transition = "background-color 0.35s ease";
    });
}

function getSavedTheme() {
    const saved = localStorage.getItem(THEME_KEY);
    return saved || "dark";
}

function toggleTheme() {
    const current = getSavedTheme();
    const next = current === "dark" ? "light" : "dark";
    localStorage.setItem(THEME_KEY, next);
    applyThemeToStreamlit(next);
}

function bindToggle() {
    const toggle = document.getElementById("theme-toggle");
    if (!toggle || toggle.dataset.bound === "true") return;
    
    toggle.dataset.bound = "true";
    toggle.addEventListener("click", toggleTheme);
    
    // Keyboard accessibility
    toggle.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            toggleTheme();
        }
    });
}

// Inject global styles for Streamlit components
function injectGlobalStyles() {
    const style = document.createElement('style');
    style.textContent = `
        /* Force transitions for smooth theme switching */
        .stApp, section[data-testid="stSidebar"], [data-testid="stChatMessage"],
        .stButton button, .stSelectbox, [data-testid="stChatInput"] textarea {
            transition: all 0.35s ease !important;
        }
        
        /* Theme-based scrollbar */
        html.theme-dark ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        html.theme-dark ::-webkit-scrollbar-track {
            background: #1A1A2E;
        }
        html.theme-dark ::-webkit-scrollbar-thumb {
            background: #6C63FF;
            border-radius: 4px;
        }
        html.theme-dark ::-webkit-scrollbar-thumb:hover {
            background: #a78bfa;
        }
        
        html.theme-light ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        html.theme-light ::-webkit-scrollbar-track {
            background: #EBEBFF;
        }
        html.theme-light ::-webkit-scrollbar-thumb {
            background: #6C63FF;
            border-radius: 4px;
        }
        html.theme-light ::-webkit-scrollbar-thumb:hover {
            background: #a78bfa;
        }
        
        /* Code blocks */
        .stCodeBlock {
            background-color: rgba(108, 99, 255, 0.1) !important;
            border-radius: 8px !important;
        }
        
        /* Success/Info/Warning/Error messages */
        .stAlert {
            background-color: rgba(108, 99, 255, 0.1) !important;
            border-left: 3px solid #6C63FF !important;
        }
    `;
    document.head.appendChild(style);
}

// Initialize and watch for Streamlit reruns
function initTheme() {
    injectGlobalStyles();
    applyThemeToStreamlit(getSavedTheme());
    bindToggle();
    
    // Watch for DOM changes (Streamlit reruns)
    const observer = new MutationObserver(() => {
        applyThemeToStreamlit(getSavedTheme());
        bindToggle();
    });
    
    observer.observe(document.body, { 
        childList: true, 
        subtree: true,
        attributes: true,
        attributeFilter: ['class', 'style']
    });
    
    // Specific observer for new chat messages
    const chatObserver = new MutationObserver(() => {
        updateChatMessages(getSavedTheme());
    });
    
    chatObserver.observe(document.body, { childList: true, subtree: true });
}

// Start the theme system
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initTheme);
} else {
    initTheme();
}