"""
llms/custom/orchestrator.py
────────────────────────────
Multi-model orchestration engine.

Workflow:
  1. Classify task + complexity
  2. Route to specialist models
  3. Synthesize final response
  4. Validate output
"""
from __future__ import annotations

import re
from typing import Optional, List, Dict, Tuple

# ── Routing Tables ──────────────────────────────────────────────────────────

# ROUTING TABLE:
# | Task Category     | Specialist Model      |
# |-------------------|-----------------------|
# | coding            | qwen2_5_coder_7b      |
# | mathematical      | deepseek_r1           |
# | reasoning         | llama-8b-instant      |
# | summarization     | gemini-2.5-flash      |
# | creative_writing  | llama-8b-instant      |
# | data_extraction   | granite3-dense-2b     |
# | validation        | granite3-dense-2b     |
# | multilingual      | qwen3.5-0.8b          |
# | conversational    | falcon3               |
# | enterprise_logic  | granite3-dense-2b     |

TASK_ROUTING: Dict[str, str] = {
    "coding":           "qwen2_5_coder_7b",
    "mathematical":     "deepseek_r1",
    "reasoning":        "llama-8b-instant",
    # "summarization":    "gemini-2.5-flash",
    "creative_writing": "llama-8b-instant",
    "data_extraction":  "granite3-dense-2b",
    "validation":       "granite3-dense-2b",
    "multilingual":     "qwen3.5-0.8b",
    "conversational":   "falcon3",
    "enterprise_logic": "granite3-dense-2b",
}

# COMPLEXITY TABLE:
# | Level    | Words     |
# |----------|-----------|
# | low      | ≤ 20      |
# | medium   | 21-80     |
# | high     | 81-200    |
# | expert   | > 200     |

COMPLEXITY_THRESHOLDS: Dict[str, int] = {
    "low":    20,
    "medium": 80,
    "high":   200,
}

# Model Keywords Table:
# | Category          | Keywords                                    |
# |-------------------|---------------------------------------------|
# | coding            | code, debug, function, class, implement... |
# | mathematical      | math, calculate, equation, solve...        |
# | reasoning         | why, explain, analyse, compare...          |
# | summarization     | summarize, summary, brief, overview...     |
# | creative_writing  | write, story, poem, essay, creative...     |
# | data_extraction   | extract, parse, json, csv, structured...   |
# | validation        | validate, verify, check, correct...        |
# | multilingual      | translate, french, spanish, german...      |
# | conversational    | hello, hi, hey, thanks, how are you...     |
# | enterprise_logic  | workflow, process, compliance, policy...   |

TASK_KEYWORDS: Dict[str, List[str]] = {
    "coding": [
        "code", "debug", "function", "class", "implement", "refactor",
        "bug", "error", "script", "program", "algorithm", "syntax",
        "python", "javascript", "typescript", "sql", "html", "css",
        "api", "library", "import", "compile", "runtime",
    ],
    "mathematical": [
        "math", "calculate", "equation", "solve", "integral", "derivative",
        "statistics", "probability", "matrix", "vector", "proof",
        "formula", "compute", "arithmetic", "algebra", "calculus",
    ],
    "reasoning": [
        "why", "explain", "analyse", "analyze", "compare", "evaluate",
        "argue", "reason", "logic", "deduce", "infer", "conclude",
        "think through", "step by step", "pros and cons",
    ],
    "summarization": [
        "summarize", "summarise", "summary", "tldr", "brief", "overview",
        "key points", "main points", "shorten", "condense", "outline",
    ],
    "creative_writing": [
        "write", "story", "poem", "essay", "creative", "draft",
        "blog", "article", "narrative", "fiction", "script",
    ],
    "data_extraction": [
        "extract", "parse", "json", "csv", "table", "structured",
        "list", "find all", "identify", "locate",
    ],
    "validation": [
        "validate", "verify", "check", "correct", "review", "audit",
        "ensure", "confirm", "test", "quality",
    ],
    "multilingual": [
        "translate", "french", "spanish", "german", "chinese", "arabic",
        "japanese", "korean", "hindi", "language",
    ],
    "conversational": [
        "hello", "hi", "hey", "thanks", "thank you", "how are you",
        "what do you think", "tell me about", "chat",
    ],
    "enterprise_logic": [
        "workflow", "process", "compliance", "policy", "business rule",
        "enterprise", "format", "structured output", "report",
    ],
}

# Constants
FALLBACK_MODEL = "falcon3"
SYNTHESIS_MODEL = "llama-8b-instant"

# Priority order for model selection (higher = more specialized)
PRIORITY_ORDER = [
    "mathematical",
    "coding",
    "validation",
    "data_extraction",
    "reasoning",
    "multilingual",
    "summarization",
    "enterprise_logic",
    "creative_writing",
    "conversational",
]


# ── Classification Functions ──────────────────────────────────────────────────

def classify_task(prompt: str) -> Tuple[List[str], str]:
    """
    Classifies a user prompt into categories and complexity level.
    """
    prompt_lower = prompt.lower()
    word_count = len(prompt.split())

    detected = []
    for category, keywords in TASK_KEYWORDS.items():
        if any(keyword in prompt_lower for keyword in keywords):
            detected.append(category)

    if not detected:
        detected = ["conversational"]

    # Determine complexity from table
    if word_count <= COMPLEXITY_THRESHOLDS["low"]:
        complexity = "low"
    elif word_count <= COMPLEXITY_THRESHOLDS["medium"]:
        complexity = "medium"
    elif word_count <= COMPLEXITY_THRESHOLDS["high"]:
        complexity = "high"
    else:
        complexity = "expert"

    return detected, complexity


def select_primary_model(categories: List[str], complexity: str) -> str:
    """
    Selects the best primary model based on detected categories.
    Uses priority ordering to choose the most specialist model.
    """
    for category in PRIORITY_ORDER:
        if category in categories:
            return TASK_ROUTING.get(category, FALLBACK_MODEL)
    return FALLBACK_MODEL


def get_routing_explanation(categories: List[str], selected_model: str) -> str:
    """
    Returns a brief explanation of why a specific model was chosen.
    """
    # Find which category triggered this selection
    for category, model in TASK_ROUTING.items():
        if model == selected_model and category in categories:
            return f"{category} → {selected_model}"

    primary_category = categories[0] if categories else "general"
    return f"{primary_category} → {selected_model}"


# ── Prompt Building Functions ─────────────────────────────────────────────────

def build_specialist_prompt(
    prompt: str,
    categories: List[str],
    complexity: str,
    primary_model: str,
) -> str:
    """
    Builds a specialist-aware prompt for the primary model.
    """
    category_str = ", ".join(categories)

    specialist_instructions = {
        "deepseek_r1": "Step-by-step reasoning with full chain of thought.",
        "llama-8b-instant": "Thorough, multi-step analysis with clear reasoning.",
        # "gemini-2.5-flash": "Concise, well-structured, clearly formatted.",
        "qwen2_5_coder_7b": "Clean, production-ready code with comments and error handling.",
        "granite3-dense-2b": "Structured, precise, factually consistent output.",
        "qwen3.5-0.8b": "High-quality response in user's language.",
        "falcon3": "Clear, conversational, and helpful.",
    }

    instruction = specialist_instructions.get(
        primary_model,
        "Provide the best possible response addressing all aspects of the request."
    )

    enhanced_prompt = (
        f"[Task: {category_str} | Complexity: {complexity}]\n"
        f"[Model: {primary_model}]\n\n"
        f"{instruction}\n\n"
        f"Request: {prompt}"
    )

    return enhanced_prompt


def build_synthesis_prompt(
    original_prompt: str,
    specialist_response: str,
    categories: List[str],
    primary_model: str,
) -> str:
    """
    Builds a synthesis prompt for refining the specialist response.
    """
    synthesis_prompt = (
        f"Refine this response into a polished final answer.\n\n"
        f"## Original Request\n{original_prompt}\n\n"
        f"## Specialist Response (from {primary_model})\n{specialist_response}\n\n"
        f"## Instructions\n"
        f"1. Remove redundancy and contradictions\n"
        f"2. Improve clarity and flow\n"
        f"3. Keep all correct content intact\n"
        f"4. Fix grammatical errors\n"
        f"5. Output ONLY the final response — do NOT append any routing info, "
        f"footers, or metadata. Those are added externally.\n"
    )

    return synthesis_prompt


def should_synthesize(categories: List[str], complexity: str) -> bool:
    """
    Determines whether to run the synthesis pass.

    NOTE: This controls synthesis only — not footer visibility.
          Use should_show_footer() for footer display logic.
          These two concerns are intentionally separated so that
          skipping synthesis never accidentally hides the footer.
    """
    # Skip synthesis for simple conversations
    if complexity == "low" and categories == ["conversational"]:
        return False
    return True


def should_show_footer() -> bool:
    """
    Determines whether to show the model routing attribution footer.

    Always returns True — the footer is unconditionally displayed
    regardless of task type, complexity, or whether synthesis ran.
    Kept as a function (rather than a bare constant) so call sites
    stay consistent and the behaviour can be toggled in one place
    if requirements ever change.
    """
    return True


def build_attribution_footer(
    categories: Optional[List[str]] = None,
    complexity: Optional[str] = None,
    primary_model: Optional[str] = None,
    synthesis_model: Optional[str] = None,
    fallback_used: bool = False
) -> str:
    """
    Builds a clean attribution footer that works well on any screen size.
    Always call this when should_show_footer() is True, independently of
    whether should_synthesize() returned True or False.
    """
    # Set defaults if parameters are None
    if categories is None:
        categories = ["conversational"]
    if complexity is None:
        complexity = "low"
    if primary_model is None:
        primary_model = "llama-8b-instant"

    # Define synthesis model locally if not available
    if synthesis_model is None:
        synthesis_model = "llama-8b-instant"

    # Resolve 'custom' to actual model using routing logic
    if primary_model == "custom":
        try:
            actual_primary = select_primary_model(categories, complexity)
        except Exception:
            actual_primary = "llama-8b-instant"  # Fallback
    else:
        actual_primary = primary_model

    # Resolve synthesis model if needed
    if synthesis_model == "custom":
        actual_synthesis = "llama-8b-instant"
    else:
        actual_synthesis = synthesis_model

    footer = "\n\n"
    footer += "---\n"
    footer += "🤖 **Model Routing**\n"
    footer += "---\n"
    footer += f"📋 Task: {', '.join(categories).upper()}\n\n"
    footer += f"⚡ Complexity: {complexity.upper()}\n\n"
    footer += f"🎯 Model: {actual_primary}\n\n"

    if actual_synthesis:
        footer += f"🔄 Refined by: {actual_synthesis}\n\n"

    if fallback_used:
        footer += f"⚠️ Fallback Active\n"

    footer += "---"

    return footer


# ── Main Orchestration Entry Point ────────────────────────────────────────────

def orchestrate(prompt: str, call_model_fn) -> str:
    """
    Main orchestration pipeline. Classifies the prompt, routes to the
    appropriate specialist model, optionally synthesizes, then appends
    exactly ONE attribution footer.

    Args:
        prompt:        The raw user prompt string.
        call_model_fn: A callable(model_name: str, prompt: str) -> str
                       that dispatches to your model backends.

    Returns:
        The final response string with a single footer appended.
    """
    fallback_used = False

    # ── Step 1: Classify ──────────────────────────────────────────────────
    categories, complexity = classify_task(prompt)

    # ── Step 2: Select primary model ─────────────────────────────────────
    primary_model = select_primary_model(categories, complexity)
    if primary_model == FALLBACK_MODEL and categories != ["conversational"]:
        fallback_used = True

    # ── Step 3: Build and run specialist prompt ───────────────────────────
    specialist_prompt = build_specialist_prompt(
        prompt, categories, complexity, primary_model)

    try:
        specialist_response = call_model_fn(primary_model, specialist_prompt)
    except Exception:
        # Hard fallback: use the fallback model if specialist call fails
        fallback_used = True
        specialist_response = call_model_fn(FALLBACK_MODEL, specialist_prompt)
        primary_model = FALLBACK_MODEL

    # ── Step 4: Optionally synthesize ────────────────────────────────────
    run_synthesis = should_synthesize(categories, complexity)
    synthesis_model_used: Optional[str] = None

    if run_synthesis:
        synthesis_prompt = build_synthesis_prompt(
            prompt, specialist_response, categories, primary_model
        )
        try:
            final_response = call_model_fn(SYNTHESIS_MODEL, synthesis_prompt)
            synthesis_model_used = SYNTHESIS_MODEL
        except Exception:
            # If synthesis fails, fall back to the raw specialist response
            final_response = specialist_response
    else:
        final_response = specialist_response

    # ── Step 5: Append exactly ONE footer ────────────────────────────────
    # This is the ONLY place build_attribution_footer() is called.
    # Never call it inside call_model_fn, streaming callbacks, or
    # any other helper — doing so produces duplicate footers.
    if should_show_footer():
        final_response += build_attribution_footer(
            categories=categories,
            complexity=complexity,
            primary_model=primary_model,
            synthesis_model=synthesis_model_used,
            fallback_used=fallback_used,
        )

    return final_response
