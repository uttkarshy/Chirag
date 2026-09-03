# Chirag Intent Engine — M2

The Intent Engine converts a user's natural-language request into a structured,
provider-independent representation of their actual intent.

It sits between the Chirag Gateway and the future Planner/Model Router.

## Core principle

Chirag must understand the user's goal before selecting a model, provider, or tool.

The Intent Engine should understand both:

1. What the user is asking Chirag to accomplish.
2. What inputs or sources the user has supplied to help accomplish it.

The Intent Engine must NOT be coupled to:

- a specific LLM
- a specific GPU
- NVIDIA NIM
- OpenRouter
- AWS
- any individual tool
- any individual file-processing implementation

All model/provider access must eventually happen through a common provider interface.

## Responsibilities

The Intent Engine is responsible for:

1. Understanding the user's request.
2. Identifying the primary goal.
3. Identifying desired output.
4. Extracting explicit requirements.
5. Extracting constraints.
6. Extracting preferences.
7. Identifying relevant context.
8. Detecting user-supplied inputs and sources.
9. Identifying the likely role of each supplied source.
10. Identifying missing information.
11. Determining whether clarification is required.
12. Producing structured intent for downstream planning.
13. Preserving the user's original intent without silently inventing requirements.

## User-supplied inputs and sources

Users may provide additional material along with their request.

Examples include:

- files
- images
- URLs
- GitHub repositories
- connected cloud documents
- other future data sources

The Intent Engine should recognize the presence and basic role of these inputs.

For example:

User request:

"Analyze this financial report and tell me whether the company looks attractive."

Supplied source:

`annual_report.pdf`

The Intent Engine should be able to represent the relationship conceptually as:

- goal: evaluate the company
- input source: annual_report.pdf
- source role: primary evidence
- desired output: investment analysis

The Intent Engine should NOT assume that the file contains information that has not been inspected.

It should also avoid asking the user for information that can reasonably be obtained from a supplied source after the appropriate source-processing tools inspect it.

## Input representation

The future Intent object should support a concept similar to:

{
  "inputs": [
    {
      "type": "file",
      "id": "...",
      "name": "annual_report.pdf",
      "role": "primary_source"
    }
  ]
}

Possible input types may eventually include:

- text
- file
- image
- url
- repository
- connected_source

The exact schema may evolve during implementation.

Input metadata should be provider-independent and should not contain assumptions about how the source will be processed.

## Source-processing separation

The Intent Engine identifies and describes supplied sources.

It does NOT directly:

- parse PDFs
- read spreadsheets
- execute code
- crawl websites
- clone repositories
- run OCR
- call external tools

Those capabilities belong to the future Source/Tool system.

The architecture should therefore remain:

User request + supplied inputs
        ↓
Intent Engine
        ↓
Structured Intent
        ↓
Source/Tool system + Planner
        ↓
Verified information
        ↓
Execution

## Clarification principle

For complex tasks, Chirag should ask clarifying questions before execution.

For trivial or sufficiently specified tasks, Chirag should avoid unnecessary questions.

The number of clarification questions should be controlled by a future question-budget mechanism.

Questions should be:

- relevant
- minimal
- non-repetitive
- decision-oriented
- easy for the user to answer

Chirag should avoid asking questions whose answers are already explicitly available in the user's request or can reasonably be obtained from supplied sources through the appropriate tools.

## Intent object

The future implementation should produce a structure conceptually similar to:

{
  "goal": "...",
  "output": "...",
  "requirements": [],
  "constraints": [],
  "preferences": [],
  "context": [],
  "inputs": [],
  "missing_information": [],
  "needs_clarification": false,
  "confidence": 0.0
}

The exact schema may evolve during implementation.

## Separation of responsibilities

Intent Engine:
"What does the user want, and what inputs have they supplied?"

Planner:
"What steps are required to accomplish it?"

Model Router:
"Which model/provider should perform each step?"

Tool system:
"Which external capabilities are required?"

Source system:
"How should supplied files, URLs, repositories, or other sources be inspected?"

Verifier:
"Did we actually accomplish the requested result?"

Memory:
"What relevant information should be retained or retrieved?"

These responsibilities must remain separated.

## Provider independence

The Intent Engine may use an LLM internally in the future, but the implementation
must access that LLM through the same provider abstraction used by Chirag.

Therefore the following should all be possible:

- local GPU model
- NVIDIA NIM
- OpenRouter
- another external provider
- future self-hosted model

No Intent Engine code should directly depend on one provider.

## Determinism and validation

Structured Intent output must be validated before being passed downstream.

Invalid or malformed intent should not silently proceed to execution.

The system should eventually support:

- schema validation
- confidence scoring
- clarification loops
- intent revision after user answers
- conversation context
- request complexity classification
- input/source classification
- source-role identification
- clarification decisions informed by available source inputs

## Non-goals for M2

M2 does NOT implement:

- model routing
- GPU orchestration
- tool execution
- web search
- code execution
- PDF extraction
- spreadsheet processing
- OCR
- repository cloning
- persistent memory
- Android automation
- production authentication
- public deployment

Those belong to later milestones.

## M2 success criteria

M2 is complete when Chirag can:

1. Accept a natural-language request.
2. Detect supplied inputs and sources.
3. Represent those inputs in structured form.
4. Convert the request into structured intent.
5. Detect important missing information.
6. Decide whether clarification is required.
7. Avoid unnecessary clarification when information is already available.
8. Ask appropriate clarification questions.
9. Incorporate the user's answers.
10. Produce a validated final Intent object.
11. Hand that object to the future Planner without selecting a provider itself.
