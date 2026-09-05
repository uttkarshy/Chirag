# Chirag Model Router — M3.1

The Model Router is responsible for analyzing an `ExecutionPlan` and determining the abstract model capability requirements needed to execute it.

## M3.1 Scope

M3.1 establishes a provider-independent capability contract and a deterministic capability analyzer.

### Invariants

1. **Provider Independence**: The contract and analyzer are completely unaware of model providers (OpenAI, Anthropic, Google, NVIDIA NIM, AWS Bedrock, etc.), API keys, GPUs, or runtimes.
2. **Plan-Driven**: Capability requirements are derived strictly from structured `ExecutionPlan` fields (`step_type`, `inputs`, `requirements`, `expected_output`), never inferred blindly from arbitrary words in the user's goal.
3. **Deterministic**: Capabilities and modalities are normalized, deduplicated, and sorted alphabetically.

### Core Capabilities

- `text_generation`
- `vision`
- `image_generation`
- `video_generation`
- `speech_to_text`
- `text_to_speech`
- `code_generation`

### Data Modalities

- `text`
- `image`
- `video`
- `audio`
