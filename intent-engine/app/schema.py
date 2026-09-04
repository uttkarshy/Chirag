from pydantic import BaseModel, Field


class InputSource(BaseModel):
    """Provider-independent description of a user-supplied source."""

    type: str = Field(..., description="Source type: text, file, image, url, repository, or connected_source.")
    id: str | None = Field(default=None, description="Stable source identifier when available.")
    name: str | None = Field(default=None, description="Human-readable source name.")
    role: str = Field(default="supporting_source", description="How the source relates to the task.")
    metadata: dict[str, str] = Field(default_factory=dict, description="Provider-independent source metadata.")


class ClarificationAnswer(BaseModel):
    """An answer supplied by the user to a clarification question."""

    question: str
    answer: str


class Intent(BaseModel):
    """Structured, provider-independent representation of a user's intent."""

    goal: str = Field(..., description="The primary outcome the user wants to achieve.")
    output: str | None = Field(default=None, description="The output or result the user expects.")
    requirements: list[str] = Field(default_factory=list, description="Explicit requirements stated by the user.")
    constraints: list[str] = Field(default_factory=list, description="Limitations or constraints stated by the user.")
    preferences: list[str] = Field(default_factory=list, description="User preferences that should influence execution.")
    context: list[str] = Field(default_factory=list, description="Relevant context supplied by the user.")
    inputs: list[InputSource] = Field(default_factory=list, description="User-supplied inputs and sources.")
    missing_information: list[str] = Field(default_factory=list, description="Important information still required.")
    clarification_questions: list[str] = Field(default_factory=list, description="Minimal questions needed before execution.")
    clarification_answers: list[ClarificationAnswer] = Field(default_factory=list, description="Answers incorporated during re-analysis.")
    needs_clarification: bool = Field(default=False, description="Whether the user must answer clarification questions.")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence in the structured intent.")


class AnalyzeRequest(BaseModel):
    """Input accepted by the Intent Engine analysis API."""

    goal: str
    output: str | None = None
    requirements: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    preferences: list[str] = Field(default_factory=list)
    context: list[str] = Field(default_factory=list)
    inputs: list[InputSource] = Field(default_factory=list)
    clarification_answers: list[ClarificationAnswer] = Field(default_factory=list)


class AnalyzeResponse(BaseModel):
    intent: Intent
    questions: list[str] = Field(default_factory=list)
