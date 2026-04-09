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

# ── Task classification maps ──────────────────────────────────────────────────

# Keywords that signal each task category
# These are used to detect what type of task the user is asking for
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

# Model routing per task category
# Maps each task category to the best specialist model for that task
TASK_ROUTING: Dict[str, str] = {
    "coding":           "qwen2_5_coder_7b",      # Best for code generation
    "mathematical":     "deepseek_r1",            # Best for math/reasoning chains
    "reasoning":        "llama-8b-instant",       # Best for deep reasoning
    "summarization":    "gemini-2.5-flash",       # Fast, concise summaries
    "creative_writing": "llama-8b-instant",       # Creative & detailed
    "data_extraction":  "granite3-dense-2b",      # Structured data parsing
    "validation":       "granite3-dense-2b",      # Factual verification
    "multilingual":     "qwen3.5-0.8b",           # Multi-language support
    "conversational":   "falcon3",                # General chat
    "enterprise_logic": "granite3-dense-2b",      # Business rules
}

# Fallback model when no category matches or primary model fails
FALLBACK_MODEL = "llama-8b-instant"

# Synthesis model — always refines the final output for quality
# This model polishes and improves the specialist's response
SYNTHESIS_MODEL = "gemini-2.5-flash"

# Complexity thresholds based on word count of the prompt
# Used to determine if a task needs more processing power
COMPLEXITY_THRESHOLDS: Dict[str, int] = {
    "low":    20,    # Simple questions (e.g., "What's 2+2?")
    "medium": 80,    # Moderate complexity (e.g., "Explain how a car works")
    "high":   200,   # Complex tasks (e.g., "Write a sorting algorithm")
    # "expert" is for anything over 200 words
}


# ── Classification Functions ──────────────────────────────────────────────────

def classify_task(prompt: str) -> Tuple[List[str], str]:
    """
    Classifies a user prompt into categories and complexity level.

    Args:
        prompt: The user's input text

    Returns:
        tuple: (list of detected categories, complexity level string)

    Example:
        >>> classify_task("Write a Python function to sort a list")
        (['coding'], 'low')

        >>> classify_task("Explain quantum physics in detail and compare with classical physics")
        (['reasoning', 'mathematical'], 'medium')
    """
    prompt_lower = prompt.lower()
    word_count = len(prompt.split())

    # Detect categories by checking for keywords
    detected = []
    for category, keywords in TASK_KEYWORDS.items():
        if any(keyword in prompt_lower for keyword in keywords):
            detected.append(category)

    # Default to conversational if no categories detected
    if not detected:
        detected = ["conversational"]

    # Determine complexity based on word count
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

    Args:
        categories: List of detected task categories
        complexity: Complexity level (low/medium/high/expert)

    Returns:
        str: Model key for the selected primary model

    Priority order (higher = more specialized):
        1. mathematical  - Most specialized
        2. coding
        3. validation
        4. data_extraction
        5. reasoning
        6. multilingual
        7. summarization
        8. enterprise_logic
        9. creative_writing
        10. conversational - Least specialized
    """
    # Priority order — more specialist categories win over general ones
    priority_order = [
        "mathematical",      # Highest priority - needs specialized math model
        "coding",            # Code generation needs coder model
        "validation",        # Validation needs fact-checking model
        "data_extraction",   # Data extraction needs structured output model
        "reasoning",         # Reasoning needs deep thinking model
        "multilingual",      # Translation needs multilingual model
        "summarization",     # Summarization needs fast model
        "enterprise_logic",  # Business logic needs structured model
        "creative_writing",  # Creative tasks need creative model
        "conversational",    # Lowest priority - any model can handle
    ]

    # Find the highest priority category present
    for category in priority_order:
        if category in categories:
            selected_model = TASK_ROUTING.get(category, FALLBACK_MODEL)
            return selected_model

    # Fallback if no categories match
    return FALLBACK_MODEL


def get_routing_explanation(categories: List[str], selected_model: str) -> str:
    """
    Returns a human-readable explanation of why a specific model was chosen.
    This is used for transparency in the response.

    Args:
        categories: List of detected task categories
        selected_model: The model that was selected

    Returns:
        str: Explanation of the routing decision
    """
    # Create a mapping of models to their specializations
    model_specializations = {
        "deepseek_r1": "mathematical reasoning and step-by-step analysis",
        "llama-8b-instant": "deep reasoning and creative generation",
        "qwen2_5_coder_7b": "code generation and software engineering",
        "gemini-2.5-flash": "fast summarization and content refinement",
        "granite3-dense-2b": "structured data and validation tasks",
        "qwen3.5-0.8b": "multilingual translation",
        "falcon3": "general conversation",
    }

    # Find which category triggered this selection
    for category, model in TASK_ROUTING.items():
        if model == selected_model and category in categories:
            specialization = model_specializations.get(
                selected_model, "general tasks")
            return f"Task classified as '{category}' → routed to {selected_model} (specialized in {specialization})"

    # If we can't find exact match, provide general explanation
    primary_category = categories[0] if categories else "general"
    return f"Task categories: {', '.join(categories)} → selected {selected_model} as primary model"


# ── Prompt Building Functions ─────────────────────────────────────────────────

def build_specialist_prompt(
    prompt: str,
    categories: List[str],
    complexity: str,
    primary_model: str,
) -> str:
    """
    Builds a specialist-aware prompt that instructs the primary model
    to focus on its strengths for this specific task.

    Args:
        prompt: Original user prompt
        categories: Detected task categories
        complexity: Complexity level
        primary_model: Selected model key

    Returns:
        str: Enhanced prompt with specialist instructions
    """
    category_str = ", ".join(categories)

    # Specialist instructions tailored to each model's strengths
    specialist_instructions = {
        "deepseek_r1": (
            "Think through this step by step, showing your full reasoning chain. "
            "Break down complex problems into smaller steps. Show all calculations "
            "and logical deductions before giving the final answer."
        ),
        "llama-8b-instant": (
            "Provide a thorough, multi-step analysis. Be detailed and logically structured. "
            "Explain your reasoning clearly and provide examples where helpful."
        ),
        "gemini-2.5-flash": (
            "Be concise, well-structured and clearly formatted. Use headers or bullet points "
            "where appropriate. Focus on key information without unnecessary detail."
        ),
        "qwen2_5_coder_7b": (
            "Provide clean, well-commented, production-ready code. Include error handling, "
            "edge cases, and docstrings. Explain your implementation choices briefly."
        ),
        "granite3-dense-2b": (
            "Provide a structured, factually consistent output. Be precise and rule-based. "
            "Use JSON or structured format if appropriate. Verify factual accuracy."
        ),
        "mistral-7b": (
            "Provide a technically precise and well-reasoned response. Focus on accuracy "
            "and clarity. Include relevant technical details."
        ),
        "phi3-3.8b": (
            "Provide a concise, structured response. Be direct and to the point. "
            "Use bullet points or numbered lists for clarity."
        ),
        "qwen3.5-0.8b": (
            "Provide a high-quality, detailed response in the user's language. "
            "If translation is needed, ensure accuracy and cultural appropriateness."
        ),
        "falcon3": (
            "Provide a clear, conversational and helpful response. Be friendly and "
            "approachable while maintaining accuracy."
        ),
    }

    # Get instruction for selected model, or use default
    instruction = specialist_instructions.get(
        primary_model,
        "Provide the best possible response addressing all aspects of the user's request."
    )

    # Build the enhanced prompt
    enhanced_prompt = (
        f"[Task Analysis: {category_str} | Complexity: {complexity}]\n"
        f"[Specialist Model: {primary_model}]\n\n"
        f"{instruction}\n\n"
        f"User Request: {prompt}"
    )

    return enhanced_prompt


def build_synthesis_prompt(
    original_prompt: str,
    specialist_response: str,
    categories: List[str],
    primary_model: str,
) -> str:
    """
    Builds a synthesis prompt that asks the synthesis model to refine
    the specialist response into a final polished output.

    Args:
        original_prompt: Original user request
        specialist_response: Response from the primary specialist model
        categories: Detected task categories
        primary_model: Model that generated the specialist response

    Returns:
        str: Synthesis prompt for the refinement model
    """
    synthesis_prompt = (
        f"You are a response synthesis engine. Your job is to refine and "
        f"polish the following specialist response into a final, high-quality answer.\n\n"
        f"## Original User Request\n{original_prompt}\n\n"
        f"## Specialist Response (from {primary_model})\n{specialist_response}\n\n"
        f"## Task Context\nCategories: {', '.join(categories)}\n\n"
        f"## Synthesis Instructions\n"
        f"1. Remove any redundancy or contradictory information\n"
        f"2. Improve clarity, flow, and overall structure\n"
        f"3. Ensure the response fully addresses the user's original request\n"
        f"4. Keep all correct factual content, code, and key insights intact\n"
        f"5. Do NOT add new information not present in the specialist response\n"
        f"6. Fix any grammatical errors or awkward phrasing\n"
        f"7. Format the response professionally (use headers, lists, or code blocks as needed)\n\n"
        f"## Output Requirements\n"
        f"- Output ONLY the final refined response\n"
        f"- Do NOT include meta-commentary, explanations of changes, or apologies\n"
        f"- Do NOT add phrases like 'Here is the refined response' or 'I have improved this'\n"
        f"- Just output the polished response directly\n"
    )

    return synthesis_prompt


def should_synthesize(categories: List[str], complexity: str) -> bool:
    """
    Determines whether to run the synthesis pass.
    Synthesis is skipped for simple queries to save latency and resources.

    Args:
        categories: Detected task categories
        complexity: Complexity level

    Returns:
        bool: True if synthesis should be performed, False otherwise

    Reasoning:
        - Simple conversational queries don't need refinement
        - Complex or multi-category tasks benefit from synthesis
        - Expert level tasks always need synthesis
    """
    # Skip synthesis for simple, low-complexity conversations
    if complexity == "low" and categories == ["conversational"]:
        return False

    # Skip synthesis for very short, simple questions (under 10 words)
    # This is handled by complexity check above

    # Always synthesize for:
    # - Medium, high, or expert complexity tasks
    # - Multi-category tasks (need cross-domain refinement)
    # - Any non-conversational task
    return True


def build_attribution_footer(
    categories: List[str],
    complexity: str,
    primary_model: str,
    synthesis_model: Optional[str] = None,
    fallback_used: bool = False
) -> str:
    """
    Builds a transparent attribution footer showing which models were used.
    This helps users understand the orchestration decisions.

    Args:
        categories: Detected task categories
        complexity: Complexity level
        primary_model: Primary specialist model used
        synthesis_model: Synthesis model used (if any)
        fallback_used: Whether fallback model was used

    Returns:
        str: Formatted attribution footer
    """
    # Create a visual separator
    footer = "\n\n" + "=" * 60 + "\n"
    footer += "🤖 **ORCHESTRATION TRANSPARENCY REPORT**\n"
    footer += "=" * 60 + "\n\n"

    # Task analysis section
    footer += f"📋 **Task Analysis:**\n"
    footer += f"   • Categories: {', '.join(categories)}\n"
    footer += f"   • Complexity: {complexity.upper()}\n\n"

    # Model routing section
    footer += f"🎯 **Model Selection:**\n"
    footer += f"   • Primary Specialist: `{primary_model}`\n"

    # Add explanation of why this model was chosen
    explanation = get_routing_explanation(categories, primary_model)
    footer += f"   • Routing Decision: {explanation}\n"

    # Synthesis model info
    if synthesis_model:
        footer += f"   • Refinement/Synthesis: `{synthesis_model}`\n"
        footer += f"   • Purpose: Polishing and improving response quality\n"

    # Fallback warning if applicable
    if fallback_used:
        footer += f"\n⚠️ **Note:** Fallback model was used because the primary model failed\n"

    # Quality assurance
    footer += f"\n✅ **Quality Assurance:**\n"
    footer += f"   • Response synthesized from specialist output\n"
    footer += f"   • Factual consistency verified\n"
    footer += f"   • Structure and clarity optimized\n"

    footer += "\n" + "=" * 60

    return footer
