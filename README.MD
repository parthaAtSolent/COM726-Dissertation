# COM726 · Dissertation — LangGraph Chatbot

## Full Project Structure

```
COM726/
│
├── .env.example                          ← Copy to .env, add GROQ_API_KEY
├── .gitignore
├── requirements.txt                      ← pip install -r requirements.txt
│
├── venv/                                 ← (existing) virtual environment
├── charts/                               ← (existing) dissertation charts
├── latex/                                ← (existing) dissertation LaTeX
│
├── llms/                                 ← ★ One folder per LLM model
│   ├── __init__.py                       ←   Central registry + build_llm()
│   └── llama_3_1_8b_instant/             ←   Llama 3.1 8B Instant (Groq)
│       ├── __init__.py
│       ├── config.py                     ←   MODEL_KEY, DISPLAY_NAME, params
│       └── factory.py                    ←   def build() → ChatGroq
│
└── langgraph_chatbot/                    ← Streamlit application
    │
    ├── main.py                           ← ★ Entry-point  (streamlit run main.py)
    │
    ├── config/
    │   ├── __init__.py
    │   └── settings.py                   ← Paths, env vars, defaults
    │
    ├── app/
    │   ├── __init__.py
    │   ├── core/
    │   │   ├── __init__.py
    │   │   └── graph.py                  ← LangGraph StateGraph + chatbot singleton
    │   ├── models/
    │   │   ├── __init__.py
    │   │   └── chat_state.py             ← ChatState, RoutingInfo, ThreadMeta
    │   └── utils/
    │       ├── __init__.py
    │       ├── conversation.py           ← load_conversation()
    │       ├── style_loader.py           ← inject_css / inject_js / load_template
    │       └── thread_service.py         ← In-memory thread CRUD
    │
    ├── static/
    │   ├── css/
    │   │   ├── chat.css                  ← Chat bubbles, input bar, animations
    │   │   └── sidebar.css               ← Thread list, model selector, button
    │   └── js/
    │       └── utils.js                  ← Auto-scroll, textarea resize
    │
    ├── templates/
    │   ├── chat.html                     ← Routing badge, empty-state HTML
    │   └── sidebar.html                  ← Branding header HTML
    │
    ├── .streamlit/
    │   └── config.toml                   ← Dark purple theme
    │
    └── data/
        └── .gitkeep                      ← chatbot.db created here at runtime
```

---

## Quick Start

```bash
# 1. Activate your existing venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install dependencies (if not already done)
pip install -r requirements.txt

# 3. Set your Groq API key
cp .env.example .env
# Edit .env → paste your free key from https://console.groq.com

# 4. Run the app
cd langgraph_chatbot
streamlit run main.py
```

---

## Adding a New Model

1. Create `llms/<model_folder>/` — e.g. `llms/gemini_2_5_flash/`
2. Add `config.py` with `MODEL_KEY`, `DISPLAY_NAME`, `ICON`, provider params
3. Add `factory.py` with `def build() -> <LangChain chat model>`
4. Add `__init__.py` exposing `build`, `MODEL_KEY`, `DISPLAY_NAME`, `ICON`
5. Import and register it in `llms/__init__.py` REGISTRY dict

**Nothing else changes.** The UI, graph, and services are all model-agnostic.

---

## ACID Compliance

| Principle   | How it's met |
|-------------|--------------|
| Atomicity   | Each thread service mutation is a single dict operation |
| Consistency | Thread IDs validated as UUID v4 before storage |
| Isolation   | `processing_message` flag prevents concurrent sends; Streamlit is single-threaded |
| Durability  | LangGraph's `SqliteSaver` writes all messages to `data/chatbot.db` |


## Run the FastAPI backend

### Navigate to your project folder
cd "C:\Users\parth\OneDrive - Southampton Solent University\COM726 - Dissertation\langgraph_chatbot"

### Run the FastAPI server
PYTHONPATH=. uvicorn api:app --host 0.0.0.0 --port 8000 --reload