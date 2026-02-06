# Golek Infrastructure

This module contains infrastructure integrations (monitoring, secrets, audit).

## Key Capabilities

* Metrics export (Prometheus)
* Secrets integration (Vault)
* Audit logging + provenance hooks

## Key Paths

* Metrics: `inference-golek/core/golek-infrastructure/src/main/java/tech/kayys/golek/infra/PrometheusMetrics.java`
* Secrets: `inference-golek/core/golek-infrastructure/src/main/java/tech/kayys/golek/infra/VaultSecretManager.java`
* Audit: `inference-golek/core/golek-infrastructure/src/main/java/tech/kayys/golek/infra/observability/`
