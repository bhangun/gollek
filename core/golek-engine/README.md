# Golek Engine (Execution)

This module implements the runtime execution engine, routing, and provider orchestration.

## Key Capabilities

* Inference orchestration + state machine
* Model routing and provider selection
* Optional quota enforcement and tenant isolation
* Observability + runtime metrics

## Key Paths

* Engine: `inference-golek/core/golek-engine/src/main/java/tech/kayys/golek/engine/inference/`
* Routing: `inference-golek/core/golek-engine/src/main/java/tech/kayys/golek/engine/routing/`
* Models: `inference-golek/core/golek-engine/src/main/java/tech/kayys/golek/engine/model/`
* Quota/Tenant: `inference-golek/core/golek-engine/src/main/java/tech/kayys/golek/engine/tenant/`
* Observability: `inference-golek/core/golek-engine/src/main/java/tech/kayys/golek/engine/observability/`

## Multi-Tenancy Toggle

By default, the engine runs in single-tenant mode and does not require `X-Tenant-ID`.

Enable enterprise multi-tenancy by adding the `tenant-golek-ext` dependency or by setting:
```
wayang.multitenancy.enabled=true
```

When enabled, API endpoints enforce tenant headers and tenant-specific routing/quota behaviors are activated.
