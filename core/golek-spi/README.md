# Golek API (Contracts)

This module defines the stable interfaces and DTOs shared by all Golek components and SDKs.

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

* Inference: `inference-golek/core/golek-spi/src/main/java/tech/kayys/golek/api/inference/`
* Providers: `inference-golek/core/golek-spi/src/main/java/tech/kayys/golek/api/provider/`
* Routing: `inference-golek/core/golek-spi/src/main/java/tech/kayys/golek/api/routing/`
* Streaming: `inference-golek/core/golek-spi/src/main/java/tech/kayys/golek/api/stream/`
* Errors: `inference-golek/core/golek-spi/src/main/java/tech/kayys/golek/api/exception/`
