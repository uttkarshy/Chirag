"""Prompt Optimization Runner for Chirag (M3.12).

ORCHESTRATION PIPELINE:
-----------------------
PromptOptimizationExperiment
            ↓
PromptOptimizationRunner
            ↓
PromptStrategyGenerator
            ↓
PromptVariant[]
            ↓
EvaluationTask (per variant, preserving base task properties: Same-Task Guarantee)
            ↓
EvaluationRunner (reuses existing ModelExecutor)
            ↓
Evaluator (reuses existing LLMJudgeEvaluator / DeterministicEvaluator)
            ↓
EvaluationScores & Summaries
            ↓
PromptStrategyAnalyzer (Reverse-Prompting foundation)
            ↓
Deterministic Ranking & Winner Selection
            ↓
PromptOptimizationResult
"""

from __future__ import annotations

import re
from typing import Any

from ..evaluation.evaluator import Evaluator
from ..evaluation.ranking import build_evaluation_summary, rank_scores
from ..evaluation.runner import EvaluationRunner
from ..evaluation.schema import (
    EvaluationCandidate,
    EvaluationDimension,
    EvaluationResult,
    EvaluationScore,
    EvaluationStatus,
    EvaluationSummary,
    EvaluationTask,
)
from ..executor import ModelExecutor
from .analyzer import PromptStrategyAnalyzer
from .generator import DefaultPromptStrategyGenerator, PromptStrategyGenerator
from .schema import (
    PromptOptimizationExperiment,
    PromptOptimizationResult,
    PromptStrategy,
    PromptVariant,
    StrategyEvaluationScore,
    StrategyObservation,
)
from .validation import (
    OptimizationError,
    sanitize_text,
    validate_prompt_experiment,
    validate_prompt_result,
)


class PromptOptimizationRunner:
    """Orchestrates prompt optimization experiments using existing execution and evaluation primitives."""

    def __init__(
        self,
        executor: ModelExecutor,
        evaluator: Evaluator,
        generator: PromptStrategyGenerator | None = None,
        analyzer: PromptStrategyAnalyzer | None = None,
    ) -> None:
        if executor is None or not (
            isinstance(executor, ModelExecutor) or callable(getattr(executor, "execute", None))
        ):
            raise TypeError(
                f"PromptOptimizationRunner requires a ModelExecutor instance, got {type(executor).__name__}."
            )

        if evaluator is None or not isinstance(evaluator, Evaluator):
            raise TypeError(
                f"PromptOptimizationRunner requires an Evaluator instance, got {type(evaluator).__name__}."
            )

        self.executor = executor
        self.evaluator = evaluator
        self.generator = generator or DefaultPromptStrategyGenerator()
        self.analyzer = analyzer or PromptStrategyAnalyzer()

    def run(
        self,
        experiment: PromptOptimizationExperiment,
    ) -> PromptOptimizationResult:
        """Run the prompt optimization experiment and return the ranked results and winner.

        Rules:
        - Same Task Guarantee: Every strategy is tested against the base task.
        - Reuses EvaluationRunner for candidate execution.
        - Reuses Evaluator for quality/semantic scoring.
        - Reuses PromptStrategyAnalyzer for structured observations.
        - Winner selection is 100% deterministic (score desc, strategy_id asc).
        - If all strategies fail, returns a controlled failure/no-winner result.
        """
        validate_prompt_experiment(experiment)

        strategy_scores: list[StrategyEvaluationScore] = []
        strategy_summaries: dict[str, EvaluationSummary] = {}
        all_observations: list[StrategyObservation] = []
        variants_by_strategy: dict[str, PromptVariant] = {}

        # 1. Execute and evaluate each prompt strategy
        for strategy in experiment.candidate_strategies:
            variant = self.generator.generate_variant(experiment.task, strategy)
            variants_by_strategy[strategy.strategy_id] = variant

            # Build variant task preserving all base task properties (Same Task Guarantee)
            variant_task = EvaluationTask(
                task_id=f"{experiment.task.task_id}::{strategy.strategy_id}",
                input=variant.prompt,
                expected_output=experiment.task.expected_output,
                input_modalities=list(experiment.task.input_modalities),
                output_modalities=list(experiment.task.output_modalities),
                structured_output=experiment.task.structured_output,
                metadata=dict(experiment.task.metadata),
            )

            # Execute candidates using existing EvaluationRunner
            runner = EvaluationRunner(self.executor)
            run_summary = runner.run(variant_task, experiment.candidate_models)

            # Evaluate execution results using existing Evaluator
            candidate_scores = self.evaluator.evaluate(
                variant_task,
                run_summary.results,
                experiment.evaluation_dimensions,
            )

            # Build formal EvaluationSummary for this strategy
            eval_summary = build_evaluation_summary(
                variant_task,
                run_summary.results,
                candidate_scores,
                metadata={"strategy_id": strategy.strategy_id},
            )
            strategy_summaries[strategy.strategy_id] = eval_summary

            # Compute strategy-level aggregated score
            strategy_score = self._compute_strategy_score(
                strategy=strategy,
                variant=variant,
                candidate_scores=candidate_scores,
                results=run_summary.results,
                dimensions=experiment.evaluation_dimensions,
            )
            strategy_scores.append(strategy_score)

            # Analyze strategy observations (Reverse-prompting foundation)
            obs = self.analyzer.analyze(
                task=experiment.task,
                strategy=strategy,
                variant=variant,
                scores=candidate_scores,
                summary=eval_summary,
            )
            all_observations.extend(obs)

        # 2. Deterministic ranking of strategies
        # Order: overall_score descending, strategy_id ascending (tie-breaker)
        ranked_strategy_scores = sorted(
            strategy_scores,
            key=lambda s: (-s.overall_score, s.strategy_id),
        )

        # 3. Deterministic Winner Selection
        winner_strategy_id: str | None = None
        winner_score: float | None = None
        winning_strategy: PromptStrategy | None = None
        optimized_prompt: str | None = None

        if ranked_strategy_scores:
            top_candidate = ranked_strategy_scores[0]
            # Verify top candidate has qualifying score and valid execution
            top_summary = strategy_summaries.get(top_candidate.strategy_id)
            has_successful_execution = (
                top_summary is not None
                and any(r.status == EvaluationStatus.SUCCESS.value for r in top_summary.results)
            )

            if top_candidate.overall_score > 0.0 or has_successful_execution:
                winner_strategy_id = top_candidate.strategy_id
                winner_score = top_candidate.overall_score
                winning_strategy = next(
                    s for s in experiment.candidate_strategies if s.strategy_id == winner_strategy_id
                )
                optimized_prompt = variants_by_strategy[winner_strategy_id].prompt

        result = PromptOptimizationResult(
            experiment_id=experiment.experiment_id,
            task_id=experiment.task.task_id,
            strategy_scores=ranked_strategy_scores,
            winner_strategy_id=winner_strategy_id,
            winner_score=winner_score,
            winning_strategy=winning_strategy,
            optimized_prompt=optimized_prompt,
            observations=all_observations,
            strategy_summaries=strategy_summaries,
            metadata=dict(experiment.metadata),
        )
        return validate_prompt_result(result)

    def _compute_strategy_score(
        self,
        strategy: PromptStrategy,
        variant: PromptVariant,
        candidate_scores: list[EvaluationScore],
        results: list[EvaluationResult],
        dimensions: list[EvaluationDimension],
    ) -> StrategyEvaluationScore:
        """Compute aggregated strategy-level performance across candidate models."""
        if not candidate_scores or all(r.status != EvaluationStatus.SUCCESS.value for r in results):
            dim_zeros = {dim.name: 0.0 for dim in dimensions}
            return StrategyEvaluationScore(
                strategy_id=strategy.strategy_id,
                variant_id=variant.variant_id,
                overall_score=0.0,
                dimension_scores=dim_zeros,
                candidate_scores=candidate_scores,
                rationale="Execution failed for all candidate models.",
                metadata={"strategy_id": strategy.strategy_id},
            )

        # Average candidate overall scores
        avg_overall = sum(s.overall_score for s in candidate_scores) / len(candidate_scores)

        # Average candidate dimension scores
        dim_totals: dict[str, float] = {}
        dim_counts: dict[str, int] = {}
        rationales: list[str] = []

        for sc in candidate_scores:
            if sc.rationale:
                rationales.append(sanitize_text(sc.rationale))
            for dim_name, val in sc.dimension_scores.items():
                dim_totals[dim_name] = dim_totals.get(dim_name, 0.0) + val
                dim_counts[dim_name] = dim_counts.get(dim_name, 0) + 1

        dim_scores = {
            dim: round(dim_totals[dim] / dim_counts[dim], 4)
            for dim in dim_totals
        }

        combined_rationale = "; ".join(rationales[:2]) if rationales else None

        return StrategyEvaluationScore(
            strategy_id=strategy.strategy_id,
            variant_id=variant.variant_id,
            overall_score=round(avg_overall, 4),
            dimension_scores=dim_scores,
            candidate_scores=candidate_scores,
            rationale=combined_rationale,
            metadata={"strategy_id": strategy.strategy_id},
        )
