"""
llms/custom/factory.py
───────────────────────
Builds and returns the CustomOrchestrator LLM wrapper.
This is a LangChain-compatible chat model that routes internally
to specialist models based on the prompt content.

The orchestrator follows a 4-step process:
1. Classify the task type and complexity
2. Route to the best specialist model
3. Optionally refine with a synthesis model
4. Add transparency metadata about which models were used
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from typing import Any, List, Optional, Dict

import llms
from llms.custom.orchestrator import (
    classify_task,
    select_primary_model,
    build_specialist_prompt,
    build_synthesis_prompt,
    should_synthesize,
    build_attribution_footer,
    SYNTHESIS_MODEL,
    FALLBACK_MODEL,
)


class CustomOrchestrator(BaseChatModel):
    """
    LangChain-compatible chat model that orchestrates multiple
    specialist models and synthesizes a final response.

    This orchestrator:
    1. Analyzes the user's request to determine task type and complexity
    2. Routes to the most appropriate specialist model
    3. Optionally refines the response with a synthesis model
    4. Provides transparency about which models were used

    Attributes:
        temperature: Controls randomness in generation (0.0 to 1.0)
        max_tokens: Maximum tokens in the response
        show_attribution: Whether to show model attribution in responses
    """

    # Required by BaseChatModel — declare as class fields
    temperature: float = 0.7  # Default temperature for generation
    max_tokens: int = 4096     # Default max tokens
    show_attribution: bool = True  # Show which models were used

    @property
    def _llm_type(self) -> str:
        """
        Returns the type identifier for this LLM.
        Required by LangChain's BaseChatModel.
        """
        return "custom_orchestrator"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Main generation method that orchestrates the multi-model workflow.

        Args:
            messages: List of chat messages (includes conversation history)
            stop: Optional stop sequences
            **kwargs: Additional arguments

        Returns:
            ChatResult containing the final synthesized response with metadata
        """

        # ── STEP 1: Extract the latest user message ─────────────────────────────
        # Find the most recent human message in the conversation
        user_prompt = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                user_prompt = msg.content
                break

        # Handle empty messages
        if not user_prompt:
            return ChatResult(generations=[
                ChatGeneration(message=AIMessage(
                    content="No message received. Please provide a question or request."
                ))
            ])

        # ── STEP 2: Classify the task ──────────────────────────────────────────
        # Determine what type of task this is and how complex
        categories, complexity = classify_task(user_prompt)

        # Select the best specialist model for this task
        primary_model_key = select_primary_model(categories, complexity)

        # Log the routing decision for debugging
        print(f"\n[Orchestrator] Processing request:")
        print(f"  - Prompt: {user_prompt[:100]}...")
        print(f"  - Categories: {categories}")
        print(f"  - Complexity: {complexity}")
        print(f"  - Selected primary model: {primary_model_key}")

        # ── STEP 3: Build and run specialist prompt ────────────────────────────
        # Create an enhanced prompt tailored to the specialist model
        specialist_prompt = build_specialist_prompt(
            user_prompt, categories, complexity, primary_model_key
        )

        # Track which models were used (for attribution)
        models_used = {
            "primary": primary_model_key,
            "synthesis": None,
            "fallback_used": False
        }

        # Run the specialist model
        specialist_response = ""
        try:
            # Build and invoke the primary model
            primary_llm = llms.build_llm(primary_model_key)
            specialist_result = primary_llm.invoke(specialist_prompt)
            specialist_response = specialist_result.content
            print(f"  ✓ Primary model '{primary_model_key}' succeeded")

        except Exception as e:
            # Primary model failed - try fallback
            print(f"  ✗ Primary model '{primary_model_key}' failed: {e}")
            print(f"  → Attempting fallback to '{FALLBACK_MODEL}'")

            try:
                fallback_llm = llms.build_llm(FALLBACK_MODEL)
                fallback_result = fallback_llm.invoke(user_prompt)
                specialist_response = fallback_result.content
                models_used["primary"] = FALLBACK_MODEL
                models_used["fallback_used"] = True
                print(f"  ✓ Fallback model '{FALLBACK_MODEL}' succeeded")

            except Exception as fallback_err:
                # Both primary and fallback failed - return error message
                print(f"  ✗ Fallback model also failed: {fallback_err}")
                error_msg = (
                    f"⚠️ Unable to process your request. Both the primary model "
                    f"({primary_model_key}) and fallback model ({FALLBACK_MODEL}) failed. "
                    f"Last error: {fallback_err}"
                )
                return ChatResult(generations=[
                    ChatGeneration(message=AIMessage(content=error_msg))
                ])

        # ── STEP 4: Optional synthesis pass ────────────────────────────────────
        # Refine the response with a synthesis model for complex tasks
        final_response = specialist_response

        if should_synthesize(categories, complexity):
            print(f"  → Running synthesis pass with '{SYNTHESIS_MODEL}'")
            try:
                # Build and invoke the synthesis model
                synthesis_llm = llms.build_llm(SYNTHESIS_MODEL)
                synthesis_prompt = build_synthesis_prompt(
                    user_prompt,
                    specialist_response,
                    categories,
                    models_used["primary"],
                )
                synthesis_result = synthesis_llm.invoke(synthesis_prompt)
                final_response = synthesis_result.content
                models_used["synthesis"] = SYNTHESIS_MODEL
                print(f"  ✓ Synthesis pass complete")

            except Exception as synth_err:
                # Synthesis failed - use specialist response as-is
                print(
                    f"  ✗ Synthesis failed, using specialist output: {synth_err}")
                # Keep final_response as specialist_response
        else:
            print(f"  → Skipping synthesis (low complexity conversational task)")

        # ── STEP 5: Add attribution metadata ───────────────────────────────────
        # Add transparency about which models were used (if enabled)
        if self.show_attribution:
            attribution_footer = build_attribution_footer(
                categories=categories,
                complexity=complexity,
                primary_model=models_used["primary"],
                synthesis_model=models_used.get("synthesis"),
                fallback_used=models_used["fallback_used"]
            )
            final_response_with_attribution = final_response + attribution_footer
        else:
            final_response_with_attribution = final_response

        # Return the final response wrapped in LangChain's ChatResult format
        return ChatResult(generations=[
            ChatGeneration(message=AIMessage(
                content=final_response_with_attribution))
        ])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Asynchronous version of _generate.
        For now, delegates to the synchronous version.

        In a production system, you would want to implement true async
        to handle multiple requests concurrently.
        """
        # Delegate to sync for now (consider implementing true async for production)
        return self._generate(messages, stop, **kwargs)

    def disable_attribution(self) -> None:
        """
        Disable the attribution footer in responses.
        Use this if you want cleaner responses without metadata.
        """
        self.show_attribution = False
        print("[Orchestrator] Attribution footer disabled")

    def enable_attribution(self) -> None:
        """
        Enable the attribution footer in responses (default behavior).
        Shows which models were used for each response.
        """
        self.show_attribution = True
        print("[Orchestrator] Attribution footer enabled")


def build(show_attribution: bool = True) -> CustomOrchestrator:
    """
    Factory function to create and return a configured CustomOrchestrator instance.

    Args:
        show_attribution: Whether to show model attribution in responses

    Returns:
        Configured CustomOrchestrator instance

    Example:
        >>> orchestrator = build(show_attribution=True)
        >>> response = orchestrator.invoke("Explain recursion")
        >>> print(response.content)  # Will show which model was used
    """
    orchestrator = CustomOrchestrator()
    orchestrator.show_attribution = show_attribution
    return orchestrator
