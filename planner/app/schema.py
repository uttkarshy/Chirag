from enum import Enum
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


class StepType(str, Enum):
    """Allowed abstract capability categories for plan steps."""

    INSPECT_SOURCE = "inspect_source"
    REASONING = "reasoning"
    GENERATION = "generation"
    VERIFICATION = "verification"


class PlanStep(BaseModel):
    """Individual step in an execution plan."""

    id: str = Field(..., description="Unique step identifier within the plan (e.g. step_1).")
    title: str = Field(..., description="Short descriptive title of the step.")
    step_type: str = Field(..., description="Capability category: inspect_source, reasoning, generation, verification.")
    inputs: list[str] = Field(default_factory=list, description="Inputs required by this step (source names or prior step IDs).")
    depends_on: list[str] = Field(default_factory=list, description="List of step IDs that must be completed prior to this step.")
    requirements: list[str] = Field(default_factory=list, description="Explicit requirements or constraints guiding this step.")
    expected_output: str = Field(..., description="Description of the artifact or intermediate result produced.")


class ExecutionPlan(BaseModel):
    """Provider-independent directed execution plan for a validated Intent."""

    plan_id: str = Field(..., description="Deterministic, unique plan identifier.")
    goal: str = Field(..., description="The primary goal from the source intent.")
    desired_output: str = Field(..., description="The expected final output from the source intent.")
    steps: list[PlanStep] = Field(..., description="Ordered list of execution plan steps.")
    estimated_complexity: str = Field(default="simple", description="Estimated complexity: simple, medium, or complex.")
    requires_tools: bool = Field(default=False, description="Whether any step requires non-LLM tools (e.g. source inspection).")
    verification_criteria: list[str] = Field(default_factory=list, description="Checklist criteria for verifying final output.")


class PlanRequest(BaseModel):
    """Input accepted by the Planner API."""

    intent: Intent


class PlanResponse(BaseModel):
    """Response returned by the Planner API."""

    plan: ExecutionPlan
