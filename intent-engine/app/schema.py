from pydantic import BaseModel, Field


class Intent(BaseModel):
    """
    Structured, provider-independent representation of a user's intent.
    """

    goal: str = Field(
        ...,
        description="The primary outcome the user wants to achieve.",
    )

    output: str | None = Field(
        default=None,
        description="The output or result the user expects.",
    )

    requirements: list[str] = Field(
        default_factory=list,
        description="Explicit requirements stated by the user.",
    )

    constraints: list[str] = Field(
        default_factory=list,
        description="Limitations or constraints stated by the user.",
    )

    preferences: list[str] = Field(
        default_factory=list,
        description="User preferences that should influence execution.",
    )

    context: list[str] = Field(
        default_factory=list,
        description="Relevant context supplied by the user.",
    )

    missing_information: list[str] = Field(
        default_factory=list,
        description="Important information still required.",
    )

    needs_clarification: bool = Field(
        default=False,
        description="Whether the user must answer clarification questions.",
    )

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence that the structured intent accurately represents the request.",
    )
