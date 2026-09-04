import pytest

from app.schema import InputSource
from app.validation import validate_intent
from app.engine import analyze_intent


def test_complete_intent_passes_validation():
    intent = analyze_intent(
        goal="Summarize the supplied report",
        output="a concise summary",
    )
    assert validate_intent(intent) == intent


def test_missing_information_requires_questions():
    intent = analyze_intent(goal="Analyze this report")
    assert intent.needs_clarification is True
    assert validate_intent(intent) == intent


def test_unsupported_input_type_is_rejected():
    intent = analyze_intent(
        goal="Summarize this source",
        output="a summary",
        inputs=[InputSource(type="unknown_source")],
    )
    with pytest.raises(ValueError, match="Unsupported input source type"):
        validate_intent(intent)


def test_inconsistent_complete_intent_is_rejected():
    intent = analyze_intent(
        goal="Summarize this source",
        output="a summary",
    )
    intent.missing_information = ["unexpected missing field"]
    with pytest.raises(ValueError, match="complete while missing information"):
        validate_intent(intent)


def test_inconsistent_clarification_state_is_rejected():
    intent = analyze_intent(goal="Summarize this source")
    intent.clarification_questions = []
    with pytest.raises(ValueError, match="without clarification questions"):
        validate_intent(intent)
