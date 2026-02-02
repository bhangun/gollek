
### ARCHITECTURE GOLEK INFERENCE SERVER

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
   - Cloud APIs (OpenAI, Anthropic, Google)
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

5. **Multi-Tenancy & Security**
   - Tenant-scoped resource quotas
   - Isolated model pools
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