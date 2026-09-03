from .schema import Intent


def analyze_intent(
    goal: str,
    output: str | None = None,
    requirements: list[str] | None = None,
    constraints: list[str] | None = None,
    preferences: list[str] | None = None,
    context: list[str] | None = None,
) -> Intent:
    """
    Build an initial structured Intent from information already provided.

    This first implementation is deterministic. It does not call an LLM,
    provider, GPU, or external service.
    """

    requirements = requirements or []
    constraints = constraints or []
    preferences = preferences or []
    context = context or []

    missing_information: list[str] = []

    if output is None:
        missing_information.append("desired output")

    if len(goal.strip()) < 5:
        missing_information.append("clearer goal")

    needs_clarification = len(missing_information) > 0

    confidence = 0.9

    if needs_clarification:
        confidence = 0.6

    return Intent(
        goal=goal.strip(),
        output=output,
        requirements=requirements,
        constraints=constraints,
        preferences=preferences,
        context=context,
        missing_information=missing_information,
        needs_clarification=needs_clarification,
        confidence=confidence,
    )


def generate_questions(intent: Intent) -> list[str]:
    """
    Generate focused clarification questions from missing information.

    This first implementation is deterministic. A future LLM-backed
    implementation can make the questions more context-aware while
    preserving the same interface.
    """

    questions: list[str] = []

    for item in intent.missing_information:
        if item == "desired output":
            questions.append(
                "What would you like Chirag to produce as the final result?"
            )

        elif item == "clearer goal":
            questions.append(
                "What would you like Chirag to accomplish?"
            )

    return questions
