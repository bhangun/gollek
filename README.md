[![Logo][gollek-logo]][gollek-logo]

<!-- Badges -->
[![Build Status](https://github.com/bhangun/gollek/actions/workflows/ci.yml/badge.svg)](https://github.com/bhangun/gollek/actions)
[![Go Report Card](https://goreportcard.com/badge/github.com/bhangun/gollek)](https://goreportcard.com/report/github.com/bhangun/gollek)
[![GitHub release](https://img.shields.io/github/v/tag/bhangun/gollek?label=release)](https://github.com/bhangun/gollek/releases)
[![License](https://img.shields.io/github/license/bhangun/gollek)](LICENSE)
[![Docker Pulls](https://img.shields.io/docker/pulls/bhangun/gollek)](https://hub.docker.com/r/bhangun/gollek)



### ARCHITECTURE GOLLEK INFERENCE SERVER

![Error Codes Doc Check](https://github.com/bhangun/gollek/actions/workflows/error-codes.yml/badge.svg)


See more for complete documentation: [https://gollek-ai.github.io]


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
   - gollek error-as-input compatibility
   - Human-in-the-loop escalation support

### Multi-Tenancy Defaults

Gollek runs in **single-tenant mode by default**. In this mode, tenant is resolved from API key and the runtime uses API key `community` when no key is provided.

To enable multi-tenancy (enterprise mode), add the `tenant-gollek-ext` module or explicitly set the config flag.

**Enable via dependency**
```xml
<dependency>
  <groupId>tech.kayys.wayang</groupId>
  <artifactId>tenant-gollek-ext</artifactId>
  <version>${project.version}</version>
</dependency>
```

**Enable via config**
```
wayang.multitenancy.enabled=true
```

When enabled, the API enforces API key authentication (`X-API-Key` or `Authorization: ApiKey <key>`) and tenant-aware features (quotas, routing preferences, and audit tags) are activated.

### Local Paths

Gollek stores models, caches, and native libraries under `~/.gollek/` by default.
Set `GOLLEK_HOME` to override this root for local deployments.

### Error Code Docs

Regenerate `docs/error-codes.md` from source:

```bash
./scripts/generate-error-codes.sh
```

Or via Make:

```bash
make error-codes
```

### BOM Usage

`inference-gollek` now publishes `tech.kayys.gollek:gollek-bom` for centralized dependency version management.

```xml
<dependencyManagement>
  <dependencies>
    <dependency>
      <groupId>tech.kayys.gollek</groupId>
      <artifactId>gollek-bom</artifactId>
      <version>1.0.0-SNAPSHOT</version>
      <type>pom</type>
      <scope>import</scope>
    </dependency>
  </dependencies>
</dependencyManagement>
```

Then consume Gollek modules without specifying versions:

```xml
<dependency>
  <groupId>tech.kayys.gollek</groupId>
  <artifactId>gollek-sdk-java-remote</artifactId>
</dependency>
```

### CI Notes

In CI, the `gollek-spi` module runs doc generation during `generate-resources`.
The Maven profile `ci-error-codes` is activated when `CI=true`.

```bash
make ci
```


# update tag to current commit

Add all changes
```bash
git add <EVERYTHING>

git commit -m "<COMMIT MESSAGE>"
git push origin main
```

And then update tag
```bash
git tag -f test-latest
git push origin :refs/tags/test-latest
git push origin test-latest
```
then watch
```bash
gh run list -R bhangun/gollek --workflow "Gollek CLI Release" --limit 5
gh run watch -R bhangun/gollek --exit-status
gh release view test-latest -R bhangun/gollek

 gh run view  -R gollek-ai/gollek-ai.github.io --log-failed  


```


[gollek-logo]: https://github.com/bhangun/repo-assets/blob/master/gollek03%404x.png
