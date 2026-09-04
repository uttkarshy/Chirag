from .schema import ExecutionPlan, Intent

ALLOWED_STEP_TYPES = {
    "inspect_source",
    "reasoning",
    "generation",
    "verification",
}


def validate_intent_for_planning(intent: Intent) -> None:
    """Validate that an intent is complete and ready for downstream execution planning."""
    if not intent.goal or not intent.goal.strip():
        raise ValueError("Cannot plan an intent with an empty goal.")

    if not intent.output or not intent.output.strip():
        raise ValueError("Cannot plan an intent with an empty desired output.")

    if intent.needs_clarification:
        raise ValueError("Cannot plan an intent that requires clarification.")

    if intent.missing_information:
        raise ValueError(
            f"Cannot plan an intent with missing information: {intent.missing_information}"
        )


def validate_execution_plan(plan: ExecutionPlan) -> ExecutionPlan:
    """Validate structural invariants, acyclicity, and provider-independence of an ExecutionPlan."""
    if not plan.steps:
        raise ValueError("Execution plan must contain at least one step.")

    seen_ids: set[str] = set()

    for index, step in enumerate(plan.steps):
        if not step.id or not step.id.strip():
            raise ValueError(f"Step at index {index} has an empty id.")

        if step.id in seen_ids:
            raise ValueError(f"Duplicate step id detected: '{step.id}'")

        if step.step_type not in ALLOWED_STEP_TYPES:
            raise ValueError(
                f"Invalid step type: '{step.step_type}'. Allowed: {sorted(ALLOWED_STEP_TYPES)}"
            )

        for dep in step.depends_on:
            if dep == step.id:
                raise ValueError(f"Step '{step.id}' cannot depend on itself.")

            if dep not in seen_ids:
                later_ids = {s.id for s in plan.steps[index + 1 :]}
                if dep in later_ids:
                    raise ValueError(
                        f"Step '{step.id}' depends on '{dep}', which is not topologically ordered before it."
                    )
                raise ValueError(f"Step '{step.id}' depends on non-existent step '{dep}'.")

        seen_ids.add(step.id)

    # Graph cycle detection verification
    adj = {s.id: list(s.depends_on) for s in plan.steps}
    visited: dict[str, int] = {node: 0 for node in adj}  # 0: unvisited, 1: visiting, 2: visited

    def check_cycle(node: str) -> None:
        visited[node] = 1
        for neighbor in adj.get(node, []):
            if neighbor in visited:
                if visited[neighbor] == 1:
                    raise ValueError(f"Circular dependency detected involving '{node}' and '{neighbor}'.")
                if visited[neighbor] == 0:
                    check_cycle(neighbor)
        visited[node] = 2

    for node in adj:
        if visited[node] == 0:
            check_cycle(node)

    return plan
