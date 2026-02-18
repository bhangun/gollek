# Gollek API (Contracts)

This module defines the stable interfaces and DTOs shared by all Gollek components and SDKs.

## Key Capabilities

* Inference request/response contracts
* Provider interfaces and routing models
* Tool and streaming payloads
* Error and observability types

## Error Codes

Docs are generated from `ErrorCode`:

```bash
./scripts/generate-error-codes.sh
```

## Key Paths

* Inference: `inference-gollek/core/gollek-spi/src/main/java/tech/kayys/gollek/api/inference/`
* Providers: `inference-gollek/core/gollek-spi/src/main/java/tech/kayys/gollek/api/provider/`
* Routing: `inference-gollek/core/gollek-spi/src/main/java/tech/kayys/gollek/api/routing/`
* Streaming: `inference-gollek/core/gollek-spi/src/main/java/tech/kayys/gollek/api/stream/`
* Errors: `inference-gollek/core/gollek-spi/src/main/java/tech/kayys/gollek/api/exception/`
