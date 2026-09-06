# Chirag

Chirag is an AI orchestration platform/control plane designed to intelligently plan, route, execute, evaluate, and eventually optimize AI workloads while keeping model providers and compute infrastructure independently replaceable.

Chirag is under active development.

## Vision

The long-term vision of Chirag separates orchestration intelligence from execution mechanics and underlying infrastructure:

```text
User
 ↓
Gateway
 ↓
Intent Engine
 ↓
Planner
 ↓
Model Requirements
 ↓
Model Router
 ↓
Provider Router
 ↓
Model Execution
 ↓
Evaluation
 ↓
Verification / Optimization
 ↓
Final Result
```

### Core Separation

- **MODEL ("What"):** What intelligence or model is used.
- **PROVIDER ("How"):** How and through which service the model is accessed.
- **COMPUTE ("Where"):** Where the workload physically runs.

*Important: Compute routing is NOT implemented yet.*

## Current Architecture

The currently implemented control-plane pipeline processes execution requests deterministically:

```text
Client
 ↓
Gateway
 ↓
Intent Engine
 ↓
Planner
 ↓
Model Requirements
 ↓
Model Router
 ↓
Model Selection
 ↓
ModelExecutionRequest
 ↓
RoutedModelExecutor
 ↓
ProviderRegistry
 ↓
ModelProvider
 ↓
ModelResult
```

### Evaluation Architecture Flow

Candidate model execution and benchmark experiments follow a separate evaluation pipeline:

```text
EvaluationExperiment
 ↓
ModelLaboratory
 ↓
EvaluationRunner
 ↓
ModelExecutor
 ↓
EvaluationResult
```

The intelligent evaluator layer is the next layer under development.

## Implemented Milestones

### M1 — Gateway

Implemented:
- lightweight HTTP gateway
- health endpoint
- models endpoint
- chat endpoint
- provider-agnostic control-plane foundation

### M2 — Intent + Planning

Implemented:
- intent validation
- clarification handling
- deterministic planner
- execution-plan contracts
- Gateway → Intent Engine → Planner integration

### M3.1 — Model Capability Contract

Implemented:
- ModelRequirements
- capability/modalities requirements
- deterministic execution-plan analysis

### M3.2 — Deterministic Model Router

Implemented:
- RegisteredModel
- ModelRegistry
- ModelRouter
- deterministic capability matching
- deterministic model ranking

### M3.3 — Model Executor Contract

Implemented:
- ModelExecutionRequest
- ModelResult
- ModelExecutor
- MockModelExecutor
- execution validation

### M3.4 — Local Execution Adapter

Implemented:
- LocalModelBackend
- InProcessBackend
- LocalModelExecutor
- provider-independent local execution abstraction

*Note: The current in-process backend is deterministic/test-oriented and is not a real GPU model runtime.*

### M3.5 — Provider Abstraction

Implemented:
- ModelProvider
- ProviderRegistry
- ProviderHealth
- RoutedModelExecutor
- LocalModelProvider
- deterministic provider resolution

*Provider routing is deliberately separate from model routing.*

### M3.6 — OpenRouter Provider

Implemented:
- OpenRouterProvider
- OpenRouterClient
- runtime API-key configuration
- OpenAI-compatible request/response normalization
- provider health checking
- safe error handling
- mocked provider tests

*Clearly stated: No API credentials are stored in Git.*

### M3.7 — Evaluation Contract

Implemented:
- EvaluationTask
- EvaluationCandidate
- EvaluationResult
- EvaluationDimension
- EvaluationScore
- EvaluationSummary
- deterministic ranking primitives

*These define the evaluation data model.*

### M3.8 — Evaluation Runner

Implemented:
- EvaluationRunner
- sequential deterministic candidate execution
- latency measurement
- result normalization
- candidate failure isolation

*The Evaluation Runner executes candidates but does not judge quality.*

### M3.9 — Multi-Model Evaluation Laboratory

Implemented:
- EvaluationExperiment
- ExperimentRunResult
- ModelLaboratory
- same-task multi-candidate experiments
- repeated experiment execution
- provider/model combination experiments

*The laboratory currently executes and collects results. It does NOT yet perform intelligent quality judgment.*

## Current Capabilities

Chirag currently has:

- Gateway
- Intent Engine
- Planner
- deterministic model capability analysis
- deterministic model routing
- model execution contracts
- local deterministic execution adapter
- provider abstraction
- Local provider
- OpenRouter provider
- evaluation contracts
- evaluation runner
- multi-model evaluation laboratory

## Provider Architecture

Provider abstraction:

```text
ModelProvider
 ├── LocalModelProvider
 ├── OpenRouterProvider
 └── future providers
```

Future providers may include:

- NVIDIA NIM
- additional external APIs
- other inference services

*(NVIDIA NIM is not implemented.)*

## Evaluation Architecture

Separation of current and future evaluation components:

### CURRENT

- **EvaluationRunner:** executes candidates and collects normalized results.
- **ModelLaboratory:** defines/runs controlled multi-model experiments.

### FUTURE

- **Evaluator:** judges output quality.
- **LLM Judge:** future mechanism for semantic evaluation.
- **Prompt Strategy Optimizer:** future mechanism for improving prompts based on experiment results.

*(These future components are not yet implemented.)*

## Compute Architecture (FUTURE)

Intended separation:

- **MODEL:** What model is used.
- **PROVIDER:** How it is accessed.
- **COMPUTE:** Where it runs.

Future compute architecture:

```text
Chirag Control Plane
 ↓
Compute Router
 ↓
AWS / GCP / Scaleway / Oracle / Local GPU
 ↓
Disposable GPU Worker
 ↓
Model Runtime
```

*Important: Compute Router and GPU lifecycle orchestration are NOT implemented yet. The goal is to keep the Chirag control plane independent of any single cloud provider.*

## Security / Secrets

- Credentials are runtime configuration.
- API keys are not stored in source code.
- Provider-specific authentication stays inside provider configuration.
- Provider errors are sanitized.
- Tests do not require real API credentials.
- External provider calls are not made during unit tests.

## Testing

Run the test suites:

```powershell
$env:PYTHONPATH = "gateway"
pytest -q gateway/tests

$env:PYTHONPATH = "planner"
pytest -q planner/tests

$env:PYTHONPATH = "intent-engine"
pytest -q intent-engine/tests

$env:PYTHONPATH = "model-router"
pytest -q model-router/tests
```

Current test results:

- `gateway`: 9 passed
- `planner`: 28 passed
- `intent-engine`: 16 passed
- `model-router`: 244 passed
- **Total:** 297 passed

## Roadmap

All roadmap items below represent future planned work unless explicitly documented as implemented above:

### M3.10 — Intelligent Evaluation Framework
- Evaluator abstraction
- deterministic evaluation
- objective scoring
- future LLM judge architecture

### M3.11 — LLM Evaluation / Judge
- semantic evaluation
- correctness
- relevance
- completeness
- instruction following

### M3.12 — Prompt Strategy Optimization
- experiment with prompt variants
- compare results
- identify better prompt strategies
- reverse-prompting research

### Future Providers
- NVIDIA NIM
- real local model runtime
- additional external providers

### Compute Layer
- Compute Capability Contract
- Compute Registry
- Compute Router
- AWS GPU workers
- GCP GPU workers
- Scaleway GPU workers
- Oracle GPU workers
- local RTX GPU workers
- disposable worker lifecycle

### Future Platform Capabilities
- persistent memory
- tools
- verification
- multimodal workflows
- agentic execution

## Design Principles

1. Provider independence
2. Compute independence
3. Deterministic core behavior where possible
4. Explicit contracts between components
5. Replaceable providers
6. Disposable compute
7. No credentials in Git
8. Testable without external services
9. Evaluation separated from execution
10. Infrastructure separated from intelligence

## Development Status

Current development status:

M1–M3.9 implemented.

M3.10+ under active development.

Chirag is not production-ready.
