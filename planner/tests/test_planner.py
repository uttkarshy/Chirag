from app.engine import create_plan
from app.schema import InputSource, Intent


def test_simple_generation_intent_produces_single_step():
    intent = Intent(
        goal="Write a concise release note",
        output="a markdown release note",
    )
    plan = create_plan(intent)

    assert plan.goal == "Write a concise release note"
    assert plan.desired_output == "a markdown release note"
    assert len(plan.steps) == 1
    assert plan.steps[0].id == "step_1"
    assert plan.steps[0].step_type == "generation"
    assert plan.steps[0].depends_on == []
    assert plan.estimated_complexity == "simple"
    assert plan.requires_tools is False


def test_supplied_source_intent_produces_inspection_and_generation_steps():
    source = InputSource(
        type="file",
        name="annual_report.pdf",
        role="primary_source",
    )
    intent = Intent(
        goal="Analyze annual report",
        output="an executive financial summary",
        inputs=[source],
    )
    plan = create_plan(intent)

    assert len(plan.steps) == 2
    assert plan.steps[0].step_type == "inspect_source"
    assert plan.steps[0].inputs == ["annual_report.pdf"]
    assert plan.steps[0].depends_on == []

    assert plan.steps[1].step_type == "generation"
    assert plan.steps[1].depends_on == ["step_1"]
    assert "step_1" in plan.steps[1].inputs

    assert plan.requires_tools is False
    assert plan.estimated_complexity == "medium"


def test_requirements_and_constraints_produce_verification_step():
    intent = Intent(
        goal="Generate API documentation",
        output="OpenAPI specification",
        requirements=["Include authentication details", "Document error responses"],
        constraints=["Max response time 200ms"],
    )
    plan = create_plan(intent)

    assert len(plan.steps) == 2
    assert plan.steps[0].step_type == "generation"

    assert plan.steps[1].step_type == "verification"
    assert plan.steps[1].depends_on == ["step_1"]
    assert "step_1" in plan.steps[1].inputs
    assert len(plan.verification_criteria) == 3


def test_multiple_sources_and_constraints_complexity():
    sources = [
        InputSource(type="file", name="doc1.pdf"),
        InputSource(type="url", name="https://example.com/api"),
        InputSource(type="repository", name="uttkarshy/Chirag"),
    ]
    intent = Intent(
        goal="Audit codebase and documentation",
        output="Security audit report",
        inputs=sources,
        requirements=["Zero external dependencies"],
        constraints=["Strict latency budget"],
    )
    plan = create_plan(intent)

    assert len(plan.steps) == 5  # 3 inspect_source + 1 generation + 1 verification
    assert plan.steps[0].step_type == "inspect_source"
    assert plan.steps[1].step_type == "inspect_source"
    assert plan.steps[2].step_type == "inspect_source"
    assert plan.steps[3].step_type == "generation"
    assert plan.steps[3].depends_on == ["step_1", "step_2", "step_3"]
    assert plan.steps[4].step_type == "verification"
    assert plan.steps[4].depends_on == ["step_4"]

    assert plan.estimated_complexity == "complex"
    assert plan.requires_tools is False


def test_requires_tools_remains_false_even_with_inspect_source_steps():
    intent = Intent(
        goal="Analyze data file",
        output="data summary",
        inputs=[InputSource(type="file", name="dataset.csv")],
    )
    plan = create_plan(intent)
    assert any(s.step_type == "inspect_source" for s in plan.steps)
    assert plan.requires_tools is False


def test_same_request_produces_same_plan_id():
    intent1 = Intent(
        goal="Generate tests",
        output="pytest suite",
        inputs=[InputSource(type="file", name="main.py")],
        requirements=["high coverage"],
        constraints=["no external network"],
    )
    intent2 = Intent(
        goal="Generate tests",
        output="pytest suite",
        inputs=[InputSource(type="file", name="main.py")],
        requirements=["high coverage"],
        constraints=["no external network"],
    )
    plan1 = create_plan(intent1)
    plan2 = create_plan(intent2)

    assert plan1.plan_id == plan2.plan_id
    assert plan1.plan_id.startswith("plan_")


def test_changing_source_identity_changes_plan_id():
    base_intent = Intent(
        goal="Process report",
        output="summary",
        inputs=[InputSource(type="file", name="report_2024.pdf")],
    )
    modified_intent = Intent(
        goal="Process report",
        output="summary",
        inputs=[InputSource(type="file", name="report_2025.pdf")],
    )
    plan1 = create_plan(base_intent)
    plan2 = create_plan(modified_intent)

    assert plan1.plan_id != plan2.plan_id


def test_changing_constraint_changes_plan_id():
    intent1 = Intent(
        goal="Build service",
        output="codebase",
        constraints=["cost under $10"],
    )
    intent2 = Intent(
        goal="Build service",
        output="codebase",
        constraints=["cost under $1000"],
    )
    plan1 = create_plan(intent1)
    plan2 = create_plan(intent2)

    assert plan1.plan_id != plan2.plan_id


def test_changing_requirement_changes_plan_id():
    intent1 = Intent(
        goal="Build service",
        output="codebase",
        requirements=["offline-only"],
    )
    intent2 = Intent(
        goal="Build service",
        output="codebase",
        requirements=["multi-region deployment"],
    )
    plan1 = create_plan(intent1)
    plan2 = create_plan(intent2)

    assert plan1.plan_id != plan2.plan_id


def test_goal_mentioning_provider_or_model_is_accepted():
    intent = Intent(
        goal="Build an AWS deployment pipeline using OpenAI and Claude models",
        output="Deployment script and documentation",
        requirements=["Zero downtime"],
    )
    plan = create_plan(intent)

    assert plan.goal == "Build an AWS deployment pipeline using OpenAI and Claude models"
    assert len(plan.steps) >= 1


def test_desired_output_mentioning_provider_or_model_is_accepted():
    intent = Intent(
        goal="Benchmark leading models",
        output="Comparative benchmark of GPT-4, Claude 3.5, and Llama 3",
    )
    plan = create_plan(intent)

    assert plan.desired_output == "Comparative benchmark of GPT-4, Claude 3.5, and Llama 3"
    assert len(plan.steps) == 1
