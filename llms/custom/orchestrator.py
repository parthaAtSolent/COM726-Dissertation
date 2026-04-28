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
        f"5. Output ONLY the final response\n"
    )

    return synthesis_prompt


def should_synthesize(categories: List[str], complexity: str) -> bool:
    """
    Determines whether to run the synthesis pass.
    """
    # Skip for simple conversations
    if complexity == "low" and categories == ["conversational"]:
        return False
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
        synthesis_model = "llama-8b-instant"  # Changed from gemini to llama

    # Resolve 'custom' to actual model using routing logic
    if primary_model == "custom":
        try:
            actual_primary = select_primary_model(categories, complexity)
        except:
            actual_primary = "llama-8b-instant"  # Fallback
    else:
        actual_primary = primary_model

    # Resolve synthesis model if needed
    if synthesis_model == "custom":
        actual_synthesis = "llama-8b-instant"  # Changed from gemini to llama
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
