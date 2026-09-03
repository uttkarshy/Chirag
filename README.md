# Chirag

Chirag is an AI orchestration platform designed to route tasks across local models, NVIDIA NIM, external model APIs, and tools.

## M1 — Gateway

The gateway is the permanent control plane. It should remain lightweight and provider-agnostic while GPU workers and model runtimes remain disposable.

### Core responsibilities

- Expose a stable HTTP API for web and Android clients.
- Manage conversations and task state.
- Route work through a provider-agnostic model gateway.
- Track provider health and capabilities.
- Keep secrets and provider-specific configuration outside application code.
- Support future ephemeral GPU workers without coupling the API to AWS.

### Planned provider types

- `local_gpu`
- `nvidia_nim`
- `openrouter`
- `external_api`

### M1 API surface

- `GET /health`
- `GET /models`
- `POST /chat`

The first implementation is intentionally small. Intent Engine, model routing, persistent memory, tools, and GPU lifecycle orchestration will be added in later milestones.
