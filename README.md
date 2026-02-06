
### ARCHITECTURE GOLEK INFERENCE SERVER

![Error Codes Doc Check](https://github.com/bhangun/golek/actions/workflows/error-codes.yml/badge.svg)

```mermaid
flowchart 

```



### ✅ What This Architecture Delivers

1. **True Plugin System**
   - First-class plugin abstraction (not just CDI beans)
   - Hot-reload capability with compatibility checks
   - Versioned plugin contracts
   - Phase-bound execution model

2. **Multi-Format Model Support**
   - GGUF (llama.cpp)
   - ONNX Runtime (CPU/CUDA/TensorRT)
   - Triton Inference Server
   - Cloud APIs (OpenAI, Anthropic, Google, Ollama)
   - Extensible provider registry

3. **Shared Runtime (Platform + Portable)**
   - Same kernel for core platform and standalone agents
   - Modular dependencies via Maven profiles
   - GraalVM native image ready
   - Minimal footprint for portable agents

4. **Production-Grade Reliability**
   - Circuit breakers and bulkheads
   - Intelligent fallback strategies
   - Warm model pools with eviction
   - Request-scoped error handling
   - Comprehensive audit trail

5. **Multi-Tenancy & Security (Optional)**
   - Tenant-scoped resource quotas (enterprise)
   - Isolated model pools (enterprise)
   - Secure credential management (Vault)
   - Row-level security

6. **Enterprise Observability**
   - OpenTelemetry distributed tracing
   - Prometheus metrics
   - Structured audit logging
   - Kafka event streaming

7. **Error Handling Integration**
   - Standardized `ErrorPayload` schema
   - Audit events for all failures
   - golek error-as-input compatibility
   - Human-in-the-loop escalation support

### Multi-Tenancy Defaults

Golek runs in **single-tenant mode by default**. In this mode, `X-Tenant-ID` is not required and the runtime uses `tenantId=default`.

To enable multi-tenancy (enterprise mode), add the `tenant-golek-ext` module or explicitly set the config flag.

**Enable via dependency**
```xml
<dependency>
  <groupId>tech.kayys.tenant</groupId>
  <artifactId>tenant-golek-ext</artifactId>
  <version>${project.version}</version>
</dependency>
```

**Enable via config**
```
wayang.multitenancy.enabled=true
```

When enabled, the API enforces `X-Tenant-ID` and tenant-aware features (quotas, routing preferences, and audit tags) are activated.

### Error Code Docs

Regenerate `docs/error-codes.md` from source:

```bash
./scripts/generate-error-codes.sh
```

Or via Make:

```bash
make error-codes
```

### CI Notes

In CI, the `golek-spi` module runs doc generation during `generate-resources`.
The Maven profile `ci-error-codes` is activated when `CI=true`.

```bash
make ci
```
