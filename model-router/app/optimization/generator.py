"""Prompt Strategy Generator for Chirag (M3.12)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..evaluation.schema import EvaluationTask
from .schema import PromptStrategy, PromptVariant
from .strategy import get_builtin_strategies
from .validation import validate_prompt_strategy, validate_prompt_variant


class PromptStrategyGenerator(ABC):
    """Abstract interface for generating prompt strategies and prompt variants."""

    @abstractmethod
    def generate_strategies(self, task: EvaluationTask) -> list[PromptStrategy]:
        """Generate or select candidate prompt strategies for the task."""
        pass

    @abstractmethod
    def generate_variant(self, task: EvaluationTask, strategy: PromptStrategy) -> PromptVariant:
        """Render a single prompt variant from a base task and strategy."""
        pass

    def generate_variants(
        self,
        task: EvaluationTask,
        strategies: list[PromptStrategy],
    ) -> list[PromptVariant]:
        """Generate prompt variants for all given strategies in stable order."""
        return [self.generate_variant(task, strat) for strat in strategies]


class DefaultPromptStrategyGenerator(PromptStrategyGenerator):
    """Deterministic, provider-neutral prompt strategy generator.

    Produces stable built-in prompt strategies and reproducible prompt variants
    without external API calls or non-deterministic generation.
    """

    def __init__(self, strategies: list[PromptStrategy] | None = None) -> None:
        if strategies is not None:
            self._strategies = [validate_prompt_strategy(s.model_copy(deep=True)) for s in strategies]
        else:
            self._strategies = get_builtin_strategies()

    def generate_strategies(self, task: EvaluationTask) -> list[PromptStrategy]:
        """Return deterministic strategies in stable order."""
        return [s.model_copy(deep=True) for s in self._strategies]

    def generate_variant(self, task: EvaluationTask, strategy: PromptStrategy) -> PromptVariant:
        """Render prompt text by substituting task input into the strategy template."""
        validate_prompt_strategy(strategy)
        rendered_prompt = strategy.template.replace("{task_input}", task.input.strip())
        variant_id = f"var_{strategy.strategy_id}_{task.task_id}"

        variant = PromptVariant(
            variant_id=variant_id,
            strategy_id=strategy.strategy_id,
            task_id=task.task_id,
            prompt=rendered_prompt,
            metadata={
                "strategy_id": strategy.strategy_id,
                "strategy_name": strategy.name,
                "task_id": task.task_id,
            },
        )
        return validate_prompt_variant(variant)
