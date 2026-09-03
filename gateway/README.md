# Chirag Gateway — M1

The gateway is Chirag's lightweight control plane. It must not depend on a GPU or a specific cloud provider.

## Design rules

1. Provider integrations live behind interfaces.
2. Secrets come from environment variables or a future secret manager.
3. The API contract must remain stable when providers change.
4. GPU workers are disposable compute, not persistent state.
5. Intent Engine and advanced routing are future milestones, not M1 responsibilities.

## Initial endpoints

### `GET /health`
Returns gateway health and version.

### `GET /models`
Returns models/providers registered with the gateway.

### `POST /chat`
Accepts a conversation request and delegates generation to the model gateway.

## Provider contract

Every provider should eventually expose a common interface for:

- generation
- streaming
- capability discovery
- health checks
- cost estimation

This allows local GPU workers, NVIDIA NIM, OpenRouter, and other providers to be swapped without changing the API layer.
