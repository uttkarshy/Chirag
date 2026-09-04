# Chirag Planner — M2.4

The Planner converts a validated, complete `Intent` object into an executable,
provider-independent `ExecutionPlan`.

It sits between the Chirag Intent Engine and the future Model Router.

## Core Principle

The Planner decomposes a validated goal into concrete execution steps before any model,
provider, GPU, or tool is selected.

The Planner is responsible for:
1. Validating that the Intent is complete and requires no clarification.
2. Generating a deterministic, provider-independent sequence or DAG of execution steps.
3. Structuring inspectable source inputs into preparatory inspection steps.
4. Structuring explicit requirements and constraints into terminal verification steps.
5. Calculating deterministic complexity estimates based on input sources and constraints.
6. Ensuring all plan steps and dependencies form an acyclic, topologically ordered graph.

The Planner must NOT:
- Select models, providers, or GPUs (reserved for M3 Model Router).
- Execute tools or parse files (reserved for Tool/Source Execution Runtime).
- Call external LLM APIs.
- Manage persistent state or databases.

## Step Types

The Planner produces steps using four abstract capabilities:
- `inspect_source`: Preparing and extracting data from user-supplied sources.
- `reasoning`: Analytical synthesis, transformation, or intermediate decision making.
- `generation`: Producing the primary desired output.
- `verification`: Validating generated output against constraints and requirements.

## API Surface

- `GET /health`: Service health check.
- `POST /plan`: Ingests a complete `Intent` and returns a validated `ExecutionPlan`.
