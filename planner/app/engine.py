import hashlib
from .schema import ExecutionPlan, InputSource, Intent, PlanStep, StepType
from .validation import validate_execution_plan, validate_intent_for_planning

INSPECTABLE_SOURCE_TYPES = {
    "file",
    "image",
    "url",
    "repository",
    "connected_source",
}


def _generate_plan_id(
    goal: str,
    output: str,
    inputs: list[InputSource],
    requirements: list[str],
    constraints: list[str],
) -> str:
    """Generate a deterministic, stateless plan identifier derived from request attributes."""
    norm_goal = goal.strip().lower()
    norm_output = output.strip().lower()
    source_keys = sorted(
        f"{src.type.strip().lower()}::{(src.id or '').strip()}::{(src.name or '').strip()}::{src.role.strip().lower()}"
        for src in inputs
    )
    norm_requirements = sorted(r.strip().lower() for r in requirements if r.strip())
    norm_constraints = sorted(c.strip().lower() for c in constraints if c.strip())

    raw = (
        f"{norm_goal}|{norm_output}|"
        f"{';'.join(source_keys)}|"
        f"{';'.join(norm_requirements)}|"
        f"{';'.join(norm_constraints)}"
    )
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return f"plan_{digest}"


def create_plan(intent: Intent) -> ExecutionPlan:
    """Build a validated, provider-independent ExecutionPlan deterministically from a complete Intent."""
    validate_intent_for_planning(intent)

    goal = intent.goal.strip()
    desired_output = (intent.output or "").strip()
    inputs = intent.inputs or []
    requirements = [r.strip() for r in intent.requirements if r.strip()]
    constraints = [c.strip() for c in intent.constraints if c.strip()]

    # Filter inspectable sources
    inspectable_sources = [src for src in inputs if src.type in INSPECTABLE_SOURCE_TYPES]

    steps: list[PlanStep] = []
    current_step_num = 1

    # Step generation: Inspection steps
    inspection_step_ids: list[str] = []
    for src in inspectable_sources:
        step_id = f"step_{current_step_num}"
        source_label = src.name or src.id or src.type
        step = PlanStep(
            id=step_id,
            title=f"Inspect source: {source_label}",
            step_type=StepType.INSPECT_SOURCE.value,
            inputs=[source_label],
            depends_on=[],
            requirements=[f"Role: {src.role}"] if src.role else [],
            expected_output=f"Extracted and validated information from {source_label}",
        )
        steps.append(step)
        inspection_step_ids.append(step_id)
        current_step_num += 1

    # Step generation: Synthesis / Reasoning / Generation
    gen_step_id = f"step_{current_step_num}"
    if inspection_step_ids:
        gen_step = PlanStep(
            id=gen_step_id,
            title=f"Synthesize evidence and generate {desired_output}",
            step_type=StepType.GENERATION.value,
            inputs=list(inspection_step_ids),
            depends_on=list(inspection_step_ids),
            requirements=list(requirements) + list(constraints),
            expected_output=desired_output,
        )
    else:
        gen_step = PlanStep(
            id=gen_step_id,
            title=f"Generate {desired_output}",
            step_type=StepType.GENERATION.value,
            inputs=[],
            depends_on=[],
            requirements=list(requirements) + list(constraints),
            expected_output=desired_output,
        )
    steps.append(gen_step)
    current_step_num += 1

    # Step generation: Terminal verification step (if requirements or constraints exist)
    verification_criteria: list[str] = []
    for r in requirements:
        verification_criteria.append(f"Satisfies requirement: {r}")
    for c in constraints:
        verification_criteria.append(f"Adheres to constraint: {c}")

    if verification_criteria:
        verify_step_id = f"step_{current_step_num}"
        verify_step = PlanStep(
            id=verify_step_id,
            title="Verify generated output against requirements and constraints",
            step_type=StepType.VERIFICATION.value,
            inputs=[gen_step_id],
            depends_on=[gen_step_id],
            requirements=list(requirements) + list(constraints),
            expected_output="Verification report confirming all requirements and constraints are satisfied",
        )
        steps.append(verify_step)
    else:
        verification_criteria.append(f"Produces expected output: {desired_output}")

    # Deterministic complexity evaluation
    total_sources = len(inspectable_sources)
    total_rules = len(requirements) + len(constraints)

    if total_sources > 2 or total_rules > 3 or (total_sources >= 2 and total_rules >= 2):
        estimated_complexity = "complex"
    elif total_sources > 0 or total_rules > 1:
        estimated_complexity = "medium"
    else:
        estimated_complexity = "simple"

    requires_tools = False

    plan_id = _generate_plan_id(goal, desired_output, inputs, requirements, constraints)

    plan = ExecutionPlan(
        plan_id=plan_id,
        goal=goal,
        desired_output=desired_output,
        steps=steps,
        estimated_complexity=estimated_complexity,
        requires_tools=requires_tools,
        verification_criteria=verification_criteria,
    )

    return validate_execution_plan(plan)
