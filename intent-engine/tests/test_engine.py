from app.engine import (
    QUESTION_BUDGET,
    analyze_intent,
    analyze_request,
    generate_questions,
    generate_questions_for_missing,
)
from app.schema import AnalyzeRequest, ClarificationAnswer, InputSource


def test_complete_intent_needs_no_clarification():
    intent = analyze_intent(
        goal="Analyze the annual report",
        output="an investment analysis",
        inputs=[InputSource(type="file", name="annual_report.pdf", role="primary_source")],
    )
    assert intent.needs_clarification is False
    assert intent.missing_information == []
    assert intent.inputs[0].name == "annual_report.pdf"
    assert generate_questions(intent) == []


def test_missing_output_generates_question():
    intent = analyze_intent(goal="Analyze this report")
    assert intent.needs_clarification is True
    assert "desired output" in intent.missing_information
    assert len(intent.clarification_questions) == 1


def test_clarification_answer_resolves_missing_output():
    request = AnalyzeRequest(
        goal="Analyze this report",
        clarification_answers=[
            ClarificationAnswer(
                question="What would you like Chirag to produce as the final result?",
                answer="A concise investment analysis",
            )
        ],
    )
    intent = analyze_request(request)
    assert intent.output == "A concise investment analysis"
    assert intent.needs_clarification is False
    assert intent.missing_information == []


def test_clarification_answer_resolves_clearer_goal():
    request = AnalyzeRequest(
        goal="Run",
        output="a report",
        clarification_answers=[
            ClarificationAnswer(
                question="What would you like Chirag to accomplish?",
                answer="Analyze annual financial metrics",
            )
        ],
    )
    intent = analyze_request(request)
    assert "clearer goal" not in intent.missing_information
    assert intent.needs_clarification is False


def test_blank_clarification_answer_does_not_resolve_missing_information():
    request = AnalyzeRequest(
        goal="Analyze this report",
        clarification_answers=[
            ClarificationAnswer(
                question="What would you like Chirag to produce as the final result?",
                answer="   ",
            )
        ],
    )
    intent = analyze_request(request)
    assert "desired output" in intent.missing_information
    assert intent.output is None
    assert intent.needs_clarification is True


def test_unrelated_clarification_does_not_resolve_missing_information():
    request = AnalyzeRequest(
        goal="Analyze this report",
        clarification_answers=[
            ClarificationAnswer(
                question="Which target environment should be used?",
                answer="production",
            )
        ],
    )
    intent = analyze_request(request)
    assert "desired output" in intent.missing_information
    assert intent.output is None
    assert intent.needs_clarification is True


def test_clarification_question_budget_remains_capped():
    assert QUESTION_BUDGET == 3
    missing = ["desired output", "clearer goal", "missing context", "missing timeline", "missing inputs"]
    questions = generate_questions_for_missing(missing)
    assert len(questions) <= QUESTION_BUDGET

    intent = analyze_intent(goal="Do")
    assert len(intent.clarification_questions) <= QUESTION_BUDGET


def test_input_metadata_is_preserved_without_source_inspection():
    source = InputSource(
        type="url",
        id="source-1",
        name="example.com/report",
        role="primary_source",
        metadata={"kind": "research"},
    )
    intent = analyze_intent(goal="Summarize the report", output="summary", inputs=[source])
    assert intent.inputs[0].type == "url"
    assert intent.inputs[0].metadata["kind"] == "research"
