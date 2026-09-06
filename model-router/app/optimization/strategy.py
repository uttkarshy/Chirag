"""Built-in Prompt Strategies for Chirag Prompt Optimization (M3.12)."""

from __future__ import annotations

from .schema import PromptStrategy
from .validation import validate_prompt_strategy

BUILTIN_STRATEGIES: list[PromptStrategy] = [
    PromptStrategy(
        strategy_id="direct",
        name="Direct Instruction",
        description="Presents the core task directly without additional framing.",
        template="{task_input}",
        metadata={"category": "baseline", "style": "concise"},
    ),
    PromptStrategy(
        strategy_id="structured",
        name="Structured Output",
        description="Instructs the model to organize the response with clear structure, sections, and bullet points.",
        template="Please address the following task in a structured format with clear headings and bullet points where appropriate:\n\n{task_input}",
        metadata={"category": "formatting", "style": "structured"},
    ),
    PromptStrategy(
        strategy_id="stepwise",
        name="Step-by-Step Reasoning",
        description="Instructs the model to approach the problem methodically and break down the answer step by step.",
        template="Please think through and address the following task step-by-step:\n\n{task_input}",
        metadata={"category": "reasoning", "style": "analytic"},
    ),
    PromptStrategy(
        strategy_id="constraint_focused",
        name="Constraint-Focused",
        description="Emphasizes strict adherence to precision, conciseness, and explicit constraints.",
        template="Provide a precise, concise, and focused response that strictly adheres to the requirements:\n\n{task_input}",
        metadata={"category": "precision", "style": "constrained"},
    ),
    PromptStrategy(
        strategy_id="expert_role",
        name="Expert Persona",
        description="Frames the task from the perspective of an expert domain specialist.",
        template="You are an expert domain specialist. Provide a comprehensive, high-quality response to the following task:\n\n{task_input}",
        metadata={"category": "role", "style": "expert"},
    ),
]


def get_builtin_strategies() -> list[PromptStrategy]:
    """Return a fresh list of all built-in deterministic prompt strategies."""
    return [validate_prompt_strategy(strat.model_copy(deep=True)) for strat in BUILTIN_STRATEGIES]


def get_builtin_strategy(strategy_id: str) -> PromptStrategy:
    """Retrieve a specific built-in strategy by ID."""
    norm_id = strategy_id.strip().lower()
    for strat in BUILTIN_STRATEGIES:
        if strat.strategy_id.lower() == norm_id:
            return validate_prompt_strategy(strat.model_copy(deep=True))
    raise KeyError(f"Unknown built-in prompt strategy '{strategy_id}'. Allowed: {[s.strategy_id for s in BUILTIN_STRATEGIES]}")
