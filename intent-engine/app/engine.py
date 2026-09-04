from .schema import AnalyzeRequest, InputSource, Intent

QUESTION_BUDGET = 3

CANONICAL_QUESTIONS = {
    "desired output": "What would you like Chirag to produce as the final result?",
    "clearer goal": "What would you like Chirag to accomplish?",
}


def _apply_clarification_answers(
    output: str | None,
    missing_information: list[str],
    answers,
) -> tuple[str | None, list[str]]:
    """Apply explicit user answers without inventing unrelated information."""
    remaining = list(missing_information)
    resolved_output = output

    canonical_output = CANONICAL_QUESTIONS["desired output"].lower()
    canonical_goal = CANONICAL_QUESTIONS["clearer goal"].lower()

    for answer in answers:
        text = answer.answer.strip()
        if not text:
            continue

        question = answer.question.strip().lower()
        if (
            (question == canonical_output or "produce as the final result" in question or "desired output" in question)
            and "desired output" in remaining
        ):
            resolved_output = text
            remaining.remove("desired output")
        elif (
            (question == canonical_goal or "accomplish" in question)
            and "clearer goal" in remaining
        ):
            remaining.remove("clearer goal")

    return resolved_output, remaining


def analyze_intent(
    goal: str,
    output: str | None = None,
    requirements: list[str] | None = None,
    constraints: list[str] | None = None,
    preferences: list[str] | None = None,
    context: list[str] | None = None,
    inputs: list[InputSource] | None = None,
    clarification_answers=None,
) -> Intent:
    """Build a validated, provider-independent Intent deterministically."""
    requirements = requirements or []
    constraints = constraints or []
    preferences = preferences or []
    context = context or []
    inputs = inputs or []
    clarification_answers = clarification_answers or []

    clean_goal = goal.strip()
    missing_information: list[str] = []

    if output is None or not output.strip():
        missing_information.append("desired output")

    if len(clean_goal) < 5:
        missing_information.append("clearer goal")

    resolved_output, missing_information = _apply_clarification_answers(
        output, missing_information, clarification_answers
    )

    questions = generate_questions_for_missing(missing_information)
    needs_clarification = bool(missing_information)

    # Confidence is deliberately conservative when required information is missing.
    confidence = 0.9 if not needs_clarification else 0.6
    if inputs:
        confidence = min(1.0, confidence + 0.05)

    return Intent(
        goal=clean_goal,
        output=resolved_output,
        requirements=list(requirements),
        constraints=list(constraints),
        preferences=list(preferences),
        context=list(context),
        inputs=list(inputs),
        missing_information=missing_information,
        clarification_questions=questions,
        clarification_answers=list(clarification_answers),
        needs_clarification=needs_clarification,
        confidence=confidence,
    )


def generate_questions_for_missing(missing_information: list[str]) -> list[str]:
    """Generate a minimal, bounded set of clarification questions."""
    questions: list[str] = []

    for item in missing_information:
        question = CANONICAL_QUESTIONS.get(item)
        if question:
            questions.append(question)

        if len(questions) >= QUESTION_BUDGET:
            break

    return questions


def generate_questions(intent: Intent) -> list[str]:
    """Compatibility wrapper for callers using the original M2 interface."""
    return generate_questions_for_missing(intent.missing_information)


def analyze_request(request: AnalyzeRequest) -> Intent:
    """Analyze the API request and incorporate any clarification answers."""
    return analyze_intent(
        goal=request.goal,
        output=request.output,
        requirements=request.requirements,
        constraints=request.constraints,
        preferences=request.preferences,
        context=request.context,
        inputs=request.inputs,
        clarification_answers=request.clarification_answers,
    )
