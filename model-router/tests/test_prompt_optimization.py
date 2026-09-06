"""Comprehensive tests for Chirag Prompt Strategy Optimization (M3.12)."""

import pytest

from app.evaluation.evaluator import DeterministicEvaluator
from app.evaluation.judge import LLMJudgeEvaluator
from app.evaluation.schema import (
    EvaluationCandidate,
    EvaluationDimension,
    EvaluationScore,
    EvaluationStatus,
    EvaluationTask,
)
from app.executor import MockModelExecutor, ModelExecutor
from app.optimization import (
    BUILTIN_STRATEGIES,
    DefaultPromptStrategyGenerator,
    InvalidPromptExperimentError,
    InvalidPromptStrategyError,
    InvalidPromptVariantError,
    OptimizationError,
    PromptOptimizationExperiment,
    PromptOptimizationResult,
    PromptOptimizationRunner,
    PromptStrategy,
    PromptStrategyAnalyzer,
    PromptStrategyGenerator,
    PromptVariant,
    StrategyEvaluationScore,
    StrategyObservation,
    get_builtin_strategies,
    get_builtin_strategy,
    sanitize_text,
    validate_prompt_experiment,
    validate_prompt_result,
    validate_prompt_strategy,
    validate_prompt_variant,
)
from app.schema import ExecutionStatus, ModelExecutionRequest, ModelResult


# ---------------------------------------------------------------------------
# Helpers & Fixtures
# ---------------------------------------------------------------------------

def make_test_task(task_id: str = "task-math-1") -> EvaluationTask:
    return EvaluationTask(
        task_id=task_id,
        input="Solve: 7 * 6",
        expected_output="42",
        input_modalities=["text"],
        output_modalities=["text"],
        metadata={"category": "arithmetic"},
    )


def make_test_candidates() -> list[EvaluationCandidate]:
    return [
        EvaluationCandidate(
            candidate_id="cand-local-1",
            model_id="mock-model",
            provider_id="local",
        ),
        EvaluationCandidate(
            candidate_id="cand-local-2",
            model_id="mock-model-2",
            provider_id="local",
        ),
    ]


def make_test_dimensions() -> list[EvaluationDimension]:
    return [
        EvaluationDimension(name="execution_success", weight=1.0),
        EvaluationDimension(name="exact_match", weight=2.0),
    ]


class ScriptedModelExecutor(ModelExecutor):
    """Predictable executor returning canned responses based on input text."""

    def __init__(self, mapping: dict[str, str] | None = None, fail_on: set[str] | None = None) -> None:
        self.mapping = mapping or {}
        self.fail_on = fail_on or set()
        self.call_history: list[ModelExecutionRequest] = []

    def execute(self, request: ModelExecutionRequest) -> ModelResult:
        self.call_history.append(request)
        for fail_keyword in self.fail_on:
            if fail_keyword in request.input:
                return ModelResult(
                    model_id=request.model_id,
                    status=ExecutionStatus.FAILED.value,
                    output=None,
                    output_modality="text",
                    error=f"Simulated execution failure for: {fail_keyword}",
                )

        output = "Default response"
        for key, val in self.mapping.items():
            if key in request.input:
                output = val
                break

        return ModelResult(
            model_id=request.model_id,
            status=ExecutionStatus.SUCCESS.value,
            output=output,
            output_modality="text",
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )


# ---------------------------------------------------------------------------
# A. PromptStrategy Tests
# ---------------------------------------------------------------------------

class TestPromptStrategyContract:
    def test_valid_strategy_creation(self):
        strat = PromptStrategy(
            strategy_id="custom_strat",
            name="Custom Strategy",
            description="A test strategy.",
            template="Answer the following: {task_input}",
            metadata={"version": "1.0"},
        )
        validated = validate_prompt_strategy(strat)
        assert validated.strategy_id == "custom_strat"
        assert validated.name == "Custom Strategy"
        assert "{task_input}" in validated.template

    def test_invalid_strategy_empty_fields(self):
        with pytest.raises(ValueError):
            PromptStrategy(strategy_id="", name="Valid", template="{task_input}")

        with pytest.raises(ValueError):
            PromptStrategy(strategy_id="s1", name="", template="{task_input}")

        with pytest.raises(ValueError):
            PromptStrategy(strategy_id="s1", name="Valid", template="")

    def test_strategy_template_missing_placeholder(self):
        strat = PromptStrategy(
            strategy_id="s1",
            name="No placeholder",
            template="Just do the task without placeholder",
        )
        with pytest.raises(InvalidPromptStrategyError, match="must contain '{task_input}'"):
            validate_prompt_strategy(strat)

    def test_strategy_immutability_and_copy(self):
        strat = get_builtin_strategy("direct")
        strat_copy = strat.model_copy(deep=True)
        assert strat_copy.strategy_id == strat.strategy_id
        assert strat_copy == strat


# ---------------------------------------------------------------------------
# B. Strategy Generation Tests
# ---------------------------------------------------------------------------

class TestStrategyGeneration:
    def test_builtin_strategies_list(self):
        strategies = get_builtin_strategies()
        assert len(strategies) == 5
        ids = [s.strategy_id for s in strategies]
        assert ids == ["direct", "structured", "stepwise", "constraint_focused", "expert_role"]

    def test_get_builtin_strategy_retrieval(self):
        strat = get_builtin_strategy("structured")
        assert strat.strategy_id == "structured"
        assert "structured" in strat.name.lower()

        # Case-insensitive
        strat_upper = get_builtin_strategy("STRUCTURED")
        assert strat_upper.strategy_id == "structured"

    def test_get_builtin_strategy_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown built-in prompt strategy"):
            get_builtin_strategy("non_existent_strat")

    def test_default_generator_determinism(self):
        task = make_test_task()
        gen = DefaultPromptStrategyGenerator()

        run1 = gen.generate_strategies(task)
        run2 = gen.generate_strategies(task)

        assert len(run1) == len(run2)
        for s1, s2 in zip(run1, run2):
            assert s1.strategy_id == s2.strategy_id
            assert s1.template == s2.template


# ---------------------------------------------------------------------------
# C. Prompt Variants Tests
# ---------------------------------------------------------------------------

class TestPromptVariants:
    def test_generate_variant_preserves_relationship(self):
        task = make_test_task()
        strategy = get_builtin_strategy("stepwise")
        gen = DefaultPromptStrategyGenerator()

        variant = gen.generate_variant(task, strategy)
        assert variant.strategy_id == "stepwise"
        assert variant.task_id == task.task_id
        assert variant.variant_id == f"var_{strategy.strategy_id}_{task.task_id}"
        assert task.input in variant.prompt
        assert "step-by-step" in variant.prompt.lower()

    def test_generate_variants_stable_order(self):
        task = make_test_task()
        strategies = get_builtin_strategies()[:3]
        gen = DefaultPromptStrategyGenerator()

        variants = gen.generate_variants(task, strategies)
        assert len(variants) == 3
        assert [v.strategy_id for v in variants] == [s.strategy_id for s in strategies]

    def test_invalid_variant_validation(self):
        with pytest.raises(ValueError):
            PromptVariant(variant_id="", strategy_id="s1", task_id="t1", prompt="p")

        with pytest.raises(ValueError):
            PromptVariant(variant_id="v1", strategy_id="", task_id="t1", prompt="p")

        with pytest.raises(ValueError):
            PromptVariant(variant_id="v1", strategy_id="s1", task_id="", prompt="p")

        with pytest.raises(ValueError):
            PromptVariant(variant_id="v1", strategy_id="s1", task_id="t1", prompt="")


# ---------------------------------------------------------------------------
# D. Experiments & Same-Task Guarantee Tests
# ---------------------------------------------------------------------------

class TestPromptOptimizationExperiment:
    def test_valid_experiment(self):
        task = make_test_task()
        candidates = make_test_candidates()
        strategies = get_builtin_strategies()[:2]
        dims = make_test_dimensions()

        exp = PromptOptimizationExperiment(
            experiment_id="exp-1",
            task=task,
            candidate_models=candidates,
            candidate_strategies=strategies,
            evaluation_dimensions=dims,
        )
        validated = validate_prompt_experiment(exp)
        assert validated.experiment_id == "exp-1"
        assert validated.task.task_id == task.task_id

    def test_experiment_empty_candidates_raises(self):
        with pytest.raises(ValueError):
            PromptOptimizationExperiment(
                experiment_id="exp-1",
                task=make_test_task(),
                candidate_models=[],
                candidate_strategies=get_builtin_strategies()[:2],
                evaluation_dimensions=make_test_dimensions(),
            )

    def test_experiment_duplicate_candidates_raises(self):
        cand = EvaluationCandidate(candidate_id="c1", model_id="m1", provider_id="local")
        with pytest.raises(ValueError, match="Duplicate candidate_id"):
            PromptOptimizationExperiment(
                experiment_id="exp-1",
                task=make_test_task(),
                candidate_models=[cand, cand],
                candidate_strategies=get_builtin_strategies()[:2],
                evaluation_dimensions=make_test_dimensions(),
            )

    def test_experiment_duplicate_strategies_raises(self):
        strat = get_builtin_strategy("direct")
        with pytest.raises(ValueError, match="Duplicate strategy_id"):
            PromptOptimizationExperiment(
                experiment_id="exp-1",
                task=make_test_task(),
                candidate_models=make_test_candidates(),
                candidate_strategies=[strat, strat],
                evaluation_dimensions=make_test_dimensions(),
            )

    def test_experiment_duplicate_dimensions_raises(self):
        dim = EvaluationDimension(name="accuracy", weight=1.0)
        with pytest.raises(ValueError, match="Duplicate dimension name"):
            PromptOptimizationExperiment(
                experiment_id="exp-1",
                task=make_test_task(),
                candidate_models=make_test_candidates(),
                candidate_strategies=get_builtin_strategies()[:2],
                evaluation_dimensions=[dim, dim],
            )


# ---------------------------------------------------------------------------
# E & F. Runner Orchestration, Scoring & Winner Selection Tests
# ---------------------------------------------------------------------------

class TestPromptOptimizationRunner:
    def test_runner_type_checks(self):
        with pytest.raises(TypeError, match="requires a ModelExecutor instance"):
            PromptOptimizationRunner(executor=None, evaluator=DeterministicEvaluator())  # type: ignore

        with pytest.raises(TypeError, match="requires an Evaluator instance"):
            PromptOptimizationRunner(executor=MockModelExecutor(), evaluator=None)  # type: ignore

    def test_runner_orchestration_and_winner_selection(self):
        # Setup: 'structured' strategy output contains '42' (exact match), 'direct' returns 'wrong'
        task = make_test_task()
        candidates = [
            EvaluationCandidate(candidate_id="cand-1", model_id="m1", provider_id="local")
        ]
        strat_direct = get_builtin_strategy("direct")
        strat_structured = get_builtin_strategy("structured")

        executor = ScriptedModelExecutor(
            mapping={
                "structured format": "42",  # From structured strategy template
                "Solve: 7 * 6": "wrong answer",  # From direct template
            }
        )
        evaluator = DeterministicEvaluator()

        runner = PromptOptimizationRunner(executor=executor, evaluator=evaluator)
        exp = PromptOptimizationExperiment(
            experiment_id="opt-exp-1",
            task=task,
            candidate_models=candidates,
            candidate_strategies=[strat_direct, strat_structured],
            evaluation_dimensions=[
                EvaluationDimension(name="execution_success", weight=1.0),
                EvaluationDimension(name="exact_match", weight=2.0),
            ],
        )

        result = runner.run(exp)

        assert result.experiment_id == "opt-exp-1"
        assert result.task_id == task.task_id
        assert len(result.strategy_scores) == 2

        # Winner should be 'structured'
        assert result.winner_strategy_id == "structured"
        assert result.winner_score is not None
        assert result.winner_score > 0.0
        assert result.winning_strategy is not None
        assert result.winning_strategy.strategy_id == "structured"
        assert "structured format" in result.optimized_prompt

    def test_runner_deterministic_tie_breaker(self):
        # Both strategies score identically: tie-breaker should choose alphabetical strategy_id
        task = make_test_task()
        candidates = [
            EvaluationCandidate(candidate_id="cand-1", model_id="m1", provider_id="local")
        ]
        strat_a = PromptStrategy(strategy_id="strat_beta", name="Beta", template="Solve: {task_input}")
        strat_b = PromptStrategy(strategy_id="strat_alpha", name="Alpha", template="Compute: {task_input}")

        executor = ScriptedModelExecutor(mapping={"Solve:": "42", "Compute:": "42"})
        evaluator = DeterministicEvaluator()

        runner = PromptOptimizationRunner(executor=executor, evaluator=evaluator)
        exp = PromptOptimizationExperiment(
            experiment_id="tie-exp",
            task=task,
            candidate_models=candidates,
            candidate_strategies=[strat_a, strat_b],
            evaluation_dimensions=[EvaluationDimension(name="exact_match", weight=1.0)],
        )

        result = runner.run(exp)
        # Scores are identical (1.0), so 'strat_alpha' must win due to alphabetical tie-breaker
        assert result.winner_strategy_id == "strat_alpha"
        assert result.strategy_scores[0].strategy_id == "strat_alpha"
        assert result.strategy_scores[1].strategy_id == "strat_beta"

    def test_runner_candidate_failure_isolation(self):
        # One candidate succeeds, one candidate fails
        task = make_test_task()
        candidates = [
            EvaluationCandidate(candidate_id="cand-ok", model_id="m-ok", provider_id="local"),
            EvaluationCandidate(candidate_id="cand-fail", model_id="m-fail", provider_id="local"),
        ]
        strat = get_builtin_strategy("direct")

        class PartialFailExecutor(ModelExecutor):
            def execute(self, request: ModelExecutionRequest) -> ModelResult:
                if request.model_id == "m-fail":
                    return ModelResult(
                        model_id=request.model_id,
                        status=ExecutionStatus.FAILED.value,
                        output_modality="text",
                        error="Simulated failure",
                    )
                return ModelResult(
                    model_id=request.model_id,
                    status=ExecutionStatus.SUCCESS.value,
                    output="42",
                    output_modality="text",
                )

        runner = PromptOptimizationRunner(
            executor=PartialFailExecutor(),
            evaluator=DeterministicEvaluator(),
        )
        exp = PromptOptimizationExperiment(
            experiment_id="fail-iso-exp",
            task=task,
            candidate_models=candidates,
            candidate_strategies=[strat],
            evaluation_dimensions=[EvaluationDimension(name="exact_match", weight=1.0)],
        )

        result = runner.run(exp)
        assert result.winner_strategy_id == "direct"
        assert len(result.strategy_summaries["direct"].results) == 2
        # One succeeded, one failed
        statuses = [r.status for r in result.strategy_summaries["direct"].results]
        assert EvaluationStatus.SUCCESS.value in statuses
        assert EvaluationStatus.FAILED.value in statuses

    def test_runner_all_failed_no_winner(self):
        task = make_test_task()
        candidates = [
            EvaluationCandidate(candidate_id="cand-1", model_id="m1", provider_id="local")
        ]
        strat = get_builtin_strategy("direct")

        class AlwaysFailExecutor(ModelExecutor):
            def execute(self, request: ModelExecutionRequest) -> ModelResult:
                return ModelResult(
                    model_id=request.model_id,
                    status=ExecutionStatus.FAILED.value,
                    output_modality="text",
                    error="All failed",
                )

        runner = PromptOptimizationRunner(
            executor=AlwaysFailExecutor(),
            evaluator=DeterministicEvaluator(),
        )
        exp = PromptOptimizationExperiment(
            experiment_id="all-fail-exp",
            task=task,
            candidate_models=candidates,
            candidate_strategies=[strat],
            evaluation_dimensions=[EvaluationDimension(name="exact_match", weight=1.0)],
        )

        result = runner.run(exp)
        assert result.winner_strategy_id is None
        assert result.winner_score is None
        assert result.winning_strategy is None
        assert result.optimized_prompt is None
        assert result.strategy_scores[0].overall_score == 0.0


# ---------------------------------------------------------------------------
# G. Reverse-Prompting Foundation Tests
# ---------------------------------------------------------------------------

class TestReversePromptingFoundation:
    def test_strategy_analyzer_observations(self):
        analyzer = PromptStrategyAnalyzer()
        task = make_test_task()
        strat = get_builtin_strategy("structured")
        variant = PromptVariant(
            variant_id="v1",
            strategy_id=strat.strategy_id,
            task_id=task.task_id,
            prompt="Structured prompt",
        )
        scores = [
            EvaluationScore(
                candidate_id="c1",
                overall_score=0.85,
                dimension_scores={"correctness": 0.90, "relevance": 0.80},
                rationale="Clear and relevant output.",
            )
        ]
        from app.evaluation.ranking import build_evaluation_summary
        summary = build_evaluation_summary(task, [], scores)

        observations = analyzer.analyze(task, strat, variant, scores, summary)
        assert len(observations) >= 3

        # Dimension scores present
        dim_names = {obs.dimension for obs in observations}
        assert "correctness" in dim_names
        assert "relevance" in dim_names
        assert "strongest_dimension" in dim_names
        assert "weakest_dimension" in dim_names

        # Verify values
        correctness_obs = next(o for o in observations if o.dimension == "correctness")
        assert correctness_obs.score == 0.90
        assert "structured" in correctness_obs.observation

    def test_analyzer_empty_scores_handled(self):
        analyzer = PromptStrategyAnalyzer()
        task = make_test_task()
        strat = get_builtin_strategy("direct")
        variant = PromptVariant(
            variant_id="v1",
            strategy_id=strat.strategy_id,
            task_id=task.task_id,
            prompt="Direct prompt",
        )
        from app.evaluation.ranking import build_evaluation_summary
        summary = build_evaluation_summary(task, [], [])

        observations = analyzer.analyze(task, strat, variant, [], summary)
        assert len(observations) == 1
        assert observations[0].dimension == "execution"
        assert observations[0].score == 0.0


# ---------------------------------------------------------------------------
# H. Security & Credential Redaction Tests
# ---------------------------------------------------------------------------

class TestSecurityAndSanitization:
    def test_sanitize_text_redacts_credentials(self):
        leaked = "Found key Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9 and sk-abcdef12345678 and api_key=supersecretkey99"
        sanitized = sanitize_text(leaked)
        assert "Bearer [REDACTED]" in sanitized
        assert "sk-[REDACTED]" in sanitized
        assert "api_key=[REDACTED]" in sanitized
        assert "supersecretkey99" not in sanitized
        assert "eyJhbG" not in sanitized

    def test_sanitization_in_analyzer_rationales(self):
        analyzer = PromptStrategyAnalyzer()
        task = make_test_task()
        strat = get_builtin_strategy("direct")
        variant = PromptVariant(
            variant_id="v1",
            strategy_id=strat.strategy_id,
            task_id=task.task_id,
            prompt="Prompt",
        )
        scores = [
            EvaluationScore(
                candidate_id="c1",
                overall_score=0.7,
                dimension_scores={"dim1": 0.7},
                rationale="Leaked credential: sk-abcdef12345678 in judge rationale.",
            )
        ]
        from app.evaluation.ranking import build_evaluation_summary
        summary = build_evaluation_summary(task, [], scores)

        observations = analyzer.analyze(task, strat, variant, scores, summary)
        qual_obs = next(o for o in observations if o.dimension == "qualitative_feedback")
        assert "sk-[REDACTED]" in qual_obs.observation
        assert "sk-abcdef12345678" not in qual_obs.observation


# ---------------------------------------------------------------------------
# I. End-to-End Integration with LLMJudgeEvaluator
# ---------------------------------------------------------------------------

class TestEndToEndWithLLMJudgeEvaluator:
    def test_prompt_optimization_with_llm_judge(self):
        """Verify prompt optimization using a real LLMJudgeEvaluator backed by a mock judge executor."""
        task = EvaluationTask(
            task_id="task-explain-quantum",
            input="Explain quantum superposition simply.",
            input_modalities=["text"],
            output_modalities=["text"],
        )
        candidates = [
            EvaluationCandidate(candidate_id="cand-model", model_id="candidate-llm", provider_id="local")
        ]
        strat_direct = get_builtin_strategy("direct")
        strat_expert = get_builtin_strategy("expert_role")

        # Mock candidate model execution
        candidate_executor = ScriptedModelExecutor(
            mapping={
                "You are an expert": "Quantum superposition is the principle that a system can exist in a linear combination of states until measured.",
                "Explain quantum": "Things can be two places at once.",
            }
        )

        # Mock Judge model execution returning structured JSON
        class MockJudgeExecutor(ModelExecutor):
            def execute(self, request: ModelExecutionRequest) -> ModelResult:
                # Judge responds with higher scores for the expert formulation
                if "linear combination" in request.input:
                    judge_json = '''{
                        "evaluations": [
                            {
                                "candidate_label": "Candidate A",
                                "scores": {
                                    "correctness": 0.95,
                                    "relevance": 0.90,
                                    "completeness": 0.90,
                                    "instruction_following": 0.95
                                },
                                "rationale": "Scientifically accurate and comprehensive explanation."
                            }
                        ]
                    }'''
                else:
                    judge_json = '''{
                        "evaluations": [
                            {
                                "candidate_label": "Candidate A",
                                "scores": {
                                    "correctness": 0.60,
                                    "relevance": 0.70,
                                    "completeness": 0.50,
                                    "instruction_following": 0.65
                                },
                                "rationale": "Overly simplistic and partially inaccurate."
                            }
                        ]
                    }'''
                return ModelResult(
                    model_id=request.model_id,
                    status=ExecutionStatus.SUCCESS.value,
                    output=judge_json,
                    output_modality="text",
                )

        judge_evaluator = LLMJudgeEvaluator(
            executor=MockJudgeExecutor(),
            judge_model_id="judge-model",
        )

        runner = PromptOptimizationRunner(
            executor=candidate_executor,
            evaluator=judge_evaluator,
        )

        exp = PromptOptimizationExperiment(
            experiment_id="quantum-opt-exp",
            task=task,
            candidate_models=candidates,
            candidate_strategies=[strat_direct, strat_expert],
            evaluation_dimensions=[
                EvaluationDimension(name="correctness", weight=1.0),
                EvaluationDimension(name="relevance", weight=1.0),
                EvaluationDimension(name="completeness", weight=1.0),
                EvaluationDimension(name="instruction_following", weight=1.0),
            ],
        )

        result = runner.run(exp)

        # Expert role strategy should clearly win due to superior judge scores
        assert result.winner_strategy_id == "expert_role"
        assert result.winner_score is not None
        assert result.winner_score > 0.90
        assert "expert domain specialist" in result.optimized_prompt
        assert len(result.observations) > 0
