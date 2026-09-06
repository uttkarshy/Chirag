"""Deterministic prompt builder for LLM Judge Evaluator (M3.11)."""

from __future__ import annotations

import json
from typing import Any

from .judge_schema import DEFAULT_JUDGE_RUBRIC
from .schema import EvaluationDimension, EvaluationTask


def get_anonymous_label(index: int) -> str:
    """Return a deterministic anonymous label for candidate at given zero-based index.

    Examples:
        0 -> 'Candidate A'
        1 -> 'Candidate B'
        25 -> 'Candidate Z'
        26 -> 'Candidate AA'
    """
    if index < 26:
        return f"Candidate {chr(ord('A') + index)}"
    first = chr(ord('A') + (index // 26) - 1)
    second = chr(ord('A') + (index % 26))
    return f"Candidate {first}{second}"


class JudgePromptBuilder:
    """Constructs deterministic, blind evaluation prompts for an LLM Judge."""

    def __init__(self, default_rubric: dict[str, str] | None = None) -> None:
        self.rubric = dict(DEFAULT_JUDGE_RUBRIC)
        if default_rubric:
            self.rubric.update(default_rubric)

    def build_prompt(
        self,
        task: EvaluationTask,
        anonymous_candidates: list[dict[str, str]],
        dimensions: list[EvaluationDimension],
        custom_rubric: dict[str, str] | None = None,
    ) -> str:
        """Build the full evaluation prompt for the LLM judge.

        Guarantees:
        - Candidate model_id and provider_id are strictly excluded (blind evaluation).
        - Evaluation criteria, task input, expected output (if present), and candidate
          outputs are formatted deterministically.
        - Output format instructions mandate valid structured JSON.
        """
        active_rubric = dict(self.rubric)
        if custom_rubric:
            active_rubric.update(custom_rubric)

        lines: list[str] = [
            "You are an impartial, expert evaluation judge.",
            "Your task is to objectively evaluate candidate outputs generated in response to the task prompt below.",
            "",
            "CRITICAL EVALUATION GUIDELINES:",
            "1. Evaluate only the quality of the outputs. Do not assume or guess which models generated them.",
            "2. Candidate labels (e.g. 'Candidate A', 'Candidate B') are anonymous and their ordering does not imply ranking.",
            "3. Score each requested dimension independently on a float scale from 0.0 (worst) to 1.0 (best).",
            "4. Base your judgment strictly on evidence present in the task input, reference output (if provided), and candidate outputs.",
            "5. Return your final evaluation as a single structured JSON object matching the requested schema exactly.",
            "",
            "=== TASK INPUT ===",
            task.input.strip(),
        ]

        if task.expected_output and task.expected_output.strip():
            lines.extend([
                "",
                "=== REFERENCE / EXPECTED OUTPUT ===",
                task.expected_output.strip(),
            ])

        lines.extend([
            "",
            "=== REQUESTED EVALUATION DIMENSIONS ===",
        ])
        for dim in dimensions:
            desc = dim.description.strip() if dim.description else active_rubric.get(dim.name, "Evaluate quality on this dimension.")
            lines.append(f"- {dim.name}: {desc}")

        lines.extend([
            "",
            "=== CANDIDATE OUTPUTS ===",
        ])
        for cand in anonymous_candidates:
            label = cand["label"]
            output = cand["output"].strip() if cand["output"] else "[No output produced]"
            lines.extend([
                f"### {label}:",
                output,
                "",
            ])

        dimension_names = [dim.name for dim in dimensions]
        schema_example = {
            "evaluations": [
                {
                    "candidate_id": "<Candidate Label, e.g. Candidate A>",
                    "scores": {dim: 0.0 for dim in dimension_names},
                    "rationale": "<Concise reasoning justifying the scores for this candidate>",
                }
            ]
        }

        lines.extend([
            "=== REQUIRED JSON RESPONSE FORMAT ===",
            "Respond ONLY with a valid JSON object matching this exact structure:",
            json.dumps(schema_example, indent=2),
            "",
            "Do NOT include any markdown code blocks, preamble, or conversational commentary outside the JSON object.",
        ])

        return "\n".join(lines)
