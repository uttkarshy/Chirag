import pytest
from app.schema import ExecutionPlan, Intent, PlanStep
from app.validation import validate_execution_plan, validate_intent_for_planning


def test_reject_intent_needing_clarification():
    intent = Intent(
        goal="Build an application",
        output="an app",
        needs_clarification=True,
        missing_information=["desired output"],
    )
    with pytest.raises(ValueError, match="requires clarification"):
        validate_intent_for_planning(intent)


def test_reject_intent_with_missing_information():
    intent = Intent(
        goal="Build an application",
        output="an app",
        needs_clarification=False,
        missing_information=["clearer goal"],
    )
    with pytest.raises(ValueError, match="missing information"):
        validate_intent_for_planning(intent)


def test_reject_intent_with_empty_goal():
    intent = Intent(goal="   ", output="an app")
    with pytest.raises(ValueError, match="empty goal"):
        validate_intent_for_planning(intent)


def test_reject_intent_with_empty_output():
    intent = Intent(goal="Build an app", output="   ")
    with pytest.raises(ValueError, match="empty desired output"):
        validate_intent_for_planning(intent)


def test_reject_plan_with_empty_steps():
    plan = ExecutionPlan(
        plan_id="plan_123",
        goal="Do something",
        desired_output="result",
        steps=[],
    )
    with pytest.raises(ValueError, match="at least one step"):
        validate_execution_plan(plan)


def test_reject_plan_with_duplicate_step_ids():
    step1 = PlanStep(id="step_1", title="Step 1", step_type="generation", expected_output="out1")
    step2 = PlanStep(id="step_1", title="Step 2", step_type="verification", expected_output="out2")
    plan = ExecutionPlan(
        plan_id="plan_123",
        goal="Do something",
        desired_output="result",
        steps=[step1, step2],
    )
    with pytest.raises(ValueError, match="Duplicate step id"):
        validate_execution_plan(plan)


def test_reject_plan_with_missing_dependencies():
    step1 = PlanStep(
        id="step_1",
        title="Step 1",
        step_type="generation",
        depends_on=["non_existent_step"],
        expected_output="out1",
    )
    plan = ExecutionPlan(
        plan_id="plan_123",
        goal="Do something",
        desired_output="result",
        steps=[step1],
    )
    with pytest.raises(ValueError, match="non-existent step"):
        validate_execution_plan(plan)


def test_reject_plan_with_circular_or_forward_dependencies():
    step1 = PlanStep(
        id="step_1",
        title="Step 1",
        step_type="generation",
        depends_on=["step_2"],
        expected_output="out1",
    )
    step2 = PlanStep(
        id="step_2",
        title="Step 2",
        step_type="verification",
        depends_on=["step_1"],
        expected_output="out2",
    )
    plan = ExecutionPlan(
        plan_id="plan_123",
        goal="Do something",
        desired_output="result",
        steps=[step1, step2],
    )
    with pytest.raises(ValueError, match="not topologically ordered"):
        validate_execution_plan(plan)


def test_reject_plan_with_invalid_step_type():
    step = PlanStep(
        id="step_1",
        title="Magic step",
        step_type="arbitrary_step_type",
        expected_output="out1",
    )
    plan = ExecutionPlan(
        plan_id="plan_123",
        goal="Do something",
        desired_output="result",
        steps=[step],
    )
    with pytest.raises(ValueError, match="Invalid step type"):
        validate_execution_plan(plan)


def test_accept_plan_with_provider_mentioned_in_user_content():
    step = PlanStep(
        id="step_1",
        title="Deploy to AWS cloud",
        step_type="generation",
        expected_output="deployment artifact",
    )
    plan = ExecutionPlan(
        plan_id="plan_123",
        goal="Deploy something to AWS using OpenAI",
        desired_output="deployment artifact",
        steps=[step],
    )
    validated = validate_execution_plan(plan)
    assert validated == plan
