"""Reverse-Prompting Foundation: Strategy Analyzer for Chirag (M3.12).

IMPORTANT ARCHITECTURAL DISTINCTION:
------------------------------------
PromptStrategyAnalyzer provides structured, deterministic observations
from evaluation results across dimensions. It does NOT claim to perform
semantic LLM-driven reverse prompting or uncontrolled recursive prompt rewriting.

It establishes the structured observation layer required for future (M3.13+)
intelligent prompt optimization.
"""

from __future__ import annotations

from typing import Any

from ..evaluation.schema import EvaluationScore, EvaluationSummary, EvaluationTask
from .schema import PromptStrategy, PromptVariant, StrategyObservation
from .validation import sanitize_text, validate_prompt_strategy, validate_prompt_variant


class PromptStrategyAnalyzer:
    """Analyzes strategy-level evaluation results and produces structured observations."""

    def analyze(
        self,
        task: EvaluationTask,
        strategy: PromptStrategy,
        variant: PromptVariant,
        scores: list[EvaluationScore],
        summary: EvaluationSummary,
    ) -> list[StrategyObservation]:
        """Derive structured strategy observations from candidate evaluation scores."""
        validate_prompt_strategy(strategy)
        validate_prompt_variant(variant)

        observations: list[StrategyObservation] = []

        if not scores:
            observations.append(
                StrategyObservation(
                    strategy_id=strategy.strategy_id,
                    dimension="execution",
                    score=0.0,
                    observation=f"Strategy '{strategy.strategy_id}' had no valid candidate evaluation scores.",
                    metadata={"task_id": task.task_id, "variant_id": variant.variant_id},
                )
            )
            return observations

        # Aggregate dimension scores across candidate models
        dim_totals: dict[str, float] = {}
        dim_counts: dict[str, int] = {}
        rationales: list[str] = []

        for sc in scores:
            if sc.rationale:
                rationales.append(sanitize_text(sc.rationale))
            for dim_name, val in sc.dimension_scores.items():
                dim_totals[dim_name] = dim_totals.get(dim_name, 0.0) + val
                dim_counts[dim_name] = dim_counts.get(dim_name, 0) + 1

        dim_averages = {
            dim: round(dim_totals[dim] / dim_counts[dim], 4)
            for dim in dim_totals
        }

        # 1. Per-dimension observations
        for dim_name, avg_score in sorted(dim_averages.items()):
            obs_text = (
                f"Strategy '{strategy.strategy_id}' achieved an average score of "
                f"{avg_score:.2f} on dimension '{dim_name}'."
            )
            observations.append(
                StrategyObservation(
                    strategy_id=strategy.strategy_id,
                    dimension=dim_name,
                    score=avg_score,
                    observation=obs_text,
                    metadata={"task_id": task.task_id, "variant_id": variant.variant_id},
                )
            )

        # 2. Strongest / weakest dimension summary observations
        if dim_averages:
            best_dim = max(dim_averages.items(), key=lambda x: x[1])
            worst_dim = min(dim_averages.items(), key=lambda x: x[1])

            observations.append(
                StrategyObservation(
                    strategy_id=strategy.strategy_id,
                    dimension="strongest_dimension",
                    score=best_dim[1],
                    observation=(
                        f"Strongest dimension for strategy '{strategy.strategy_id}' was "
                        f"'{best_dim[0]}' with score {best_dim[1]:.2f}."
                    ),
                    metadata={"task_id": task.task_id, "dimension_name": best_dim[0]},
                )
            )

            observations.append(
                StrategyObservation(
                    strategy_id=strategy.strategy_id,
                    dimension="weakest_dimension",
                    score=worst_dim[1],
                    observation=(
                        f"Weakest dimension for strategy '{strategy.strategy_id}' was "
                        f"'{worst_dim[0]}' with score {worst_dim[1]:.2f}."
                    ),
                    metadata={"task_id": task.task_id, "dimension_name": worst_dim[0]},
                )
            )

        # 3. Candidate execution failure alert observation
        failed_results = [r for r in summary.results if r.status != "success"]
        if failed_results:
            observations.append(
                StrategyObservation(
                    strategy_id=strategy.strategy_id,
                    dimension="execution_stability",
                    score=0.0,
                    observation=(
                        f"Execution instability observed: {len(failed_results)} of "
                        f"{len(summary.results)} candidate(s) failed."
                    ),
                    metadata={"failed_candidates": str([r.candidate_id for r in failed_results])},
                )
            )

        # 4. Synthesized judge rationale observation if present
        if rationales:
            combined_notes = "; ".join(rationales[:3])
            observations.append(
                StrategyObservation(
                    strategy_id=strategy.strategy_id,
                    dimension="qualitative_feedback",
                    score=round(sum(s.overall_score for s in scores) / len(scores), 4),
                    observation=f"Qualitative feedback: {combined_notes}",
                    metadata={"task_id": task.task_id},
                )
            )

        return observations
