from .schema import Intent


ALLOWED_INPUT_TYPES = {
    "text",
    "file",
    "image",
    "url",
    "repository",
    "connected_source",
}


def validate_intent(intent: Intent) -> Intent:
    """Validate semantic invariants after deterministic intent analysis.

    Pydantic validates the shape and types of the response. This layer validates
    rules that depend on the meaning and consistency of multiple fields.
    """
    if not intent.goal.strip():
        raise ValueError("Intent goal must not be empty.")

    if intent.needs_clarification:
        if not intent.missing_information:
            raise ValueError(
                "Intent cannot require clarification without missing information."
            )
        if not intent.clarification_questions:
            raise ValueError(
                "Intent cannot require clarification without clarification questions."
            )
    else:
        if intent.missing_information:
            raise ValueError(
                "Intent cannot be complete while missing information is present."
            )
        if intent.clarification_questions:
            raise ValueError(
                "Complete intent must not contain clarification questions."
            )

    if not 0.0 <= intent.confidence <= 1.0:
        raise ValueError("Intent confidence must be between 0 and 1.")

    for source in intent.inputs:
        if source.type not in ALLOWED_INPUT_TYPES:
            raise ValueError(f"Unsupported input source type: {source.type}")

    return intent
