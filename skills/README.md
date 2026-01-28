# Inference-Golek Skills

Executable capability packages for inference-golek, built on the [agent skills standard](https://agentskills.io).

Skills capture step-by-step instructions for AI agents to accomplish inference tasks using inference-golek.

## Quick Start Skills

Each skill folder contains:
- `SKILL.md` - Instructions and metadata
- `scripts/` - Executable code examples
- `references/` - Supporting documentation

### Available Skills

#### 1. [Run Inference](./run-inference/)
Execute ML model inference with support for multiple formats (GGUF, ONNX, Triton, Cloud APIs).

**When to use**: Running inference on any ML model
**Key features**: Multi-format support, circuit breakers, rate limiting, tenant isolation

#### 2. [Load Model from Repository](./load-model-from-repository/)
Load models from HuggingFace, local filesystem, or cloud storage with intelligent caching.

**When to use**: Loading models from various sources
**Key features**: HuggingFace integration, local caching, hardware detection, model versioning

#### 3. [Configure Plugin](./configure-plugin/)
Build and deploy custom plugins to extend inference-golek (model router, quota, content safety, etc.).

**When to use**: Custom inference logic or constraints
**Key features**: Model routing, quota enforcement, content filtering, plugin lifecycle management

#### 4. [Handle Multi-Tenancy](./handle-multi-tenancy/)
Implement isolated tenants with per-tenant quotas, model pools, and audit trails.

**When to use**: Multi-tenant inference service
**Key features**: Tenant isolation, resource quotas, credential separation, cost tracking, audit logging

#### 5. [Monitor Inference](./monitor-inference/)
Observe inference execution with OpenTelemetry tracing, Prometheus metrics, structured logging, and Kafka events.

**When to use**: Understanding inference behavior and debugging
**Key features**: Distributed tracing, metrics, event streaming, structured logging, Grafana integration

## Skill Structure

```
inference-golek/skills/
├── run-inference/
│   ├── SKILL.md                 # Instructions
│   ├── scripts/                 # Code examples
│   └── references/              # Additional docs
├── load-model-from-repository/
│   ├── SKILL.md
│   ├── scripts/
│   └── references/
├── configure-plugin/
│   ├── SKILL.md
│   ├── scripts/
│   └── references/
├── handle-multi-tenancy/
│   ├── SKILL.md
│   ├── scripts/
│   └── references/
├── monitor-inference/
│   ├── SKILL.md
│   ├── scripts/
│   └── references/
└── README.md (this file)
```

## Using Skills with AI Agents

Skills are designed for AI agents (like Claude, GPT, etc.) to understand and follow.

### Explicit Skill Invocation

Ask the agent to use a specific skill:

```
"Use the $run-inference skill to execute a sentiment analysis model"
```

### Implicit Skill Invocation

Describe what you need; agent picks appropriate skill:

```
"Run inference on a customer review to determine sentiment"
```

The agent will automatically select `$run-inference` skill.

## Skills Progression Path

```
┌─────────────────────────────────┐
│ Start Here: Run Inference       │  Basic inference execution
├─────────────────────────────────┤
│ Load Model from Repository      │  Where to get models
├─────────────────────────────────┤
│ Monitor Inference               │  Understand performance
├─────────────────────────────────┤
│ Handle Multi-Tenancy            │  Production isolation
├─────────────────────────────────┤
│ Configure Plugin                │  Advanced customization
└─────────────────────────────────┘
```

## Common Workflows

### Basic Inference
1. Use `run-inference` to execute a model
2. Use `monitor-inference` to track performance

### Multi-Model Ensemble
1. Use `load-model-from-repository` to get multiple models
2. Use `configure-plugin` to create model router
3. Use `run-inference` for each model
4. Use `monitor-inference` to compare

### Multi-Tenant Service
1. Use `handle-multi-tenancy` to set up isolation
2. Use `run-inference` with tenant context
3. Use `monitor-inference` for tenant-specific metrics
4. Use `configure-plugin` for custom quotas

## Core Capabilities

All skills leverage these inference-golek capabilities:

- **Multi-Format Models**: GGUF (llama.cpp), ONNX, Triton, Cloud APIs
- **Production Ready**: Circuit breakers, rate limiting, warm model pools
- **Enterprise Features**: Multi-tenancy, audit trails, cost tracking
- **Observability**: OpenTelemetry, Prometheus, structured logging
- **Extensibility**: Plugin system for custom logic

## Key Concepts

### Provider
A provider abstracts different ML inference backends (OpenAI, local ONNX, Triton, etc.).

### Model Repository
Manages model loading, caching, and versioning from multiple sources.

### Tenant Context
Isolates requests per tenant with quotas and audit trails.

### Plugin
Extends platform behavior without modifying core code.

## Example: Complete Inference Pipeline

```java
// 1. Load model (Load Model from Repository skill)
ModelRepository repo = ModelRepositoryRegistry
  .getRepository("huggingface");
Model model = repo.load("bert-base-uncased");

// 2. Create tenant context (Handle Multi-Tenancy skill)
TenantContext tenant = TenantContext.builder()
  .tenantId("customer-1")
  .userId("user-456")
  .build();

// 3. Run inference (Run Inference skill)
ProviderRequest request = ProviderRequest.builder()
  .tenantContext(tenant)
  .modelId("bert-base-uncased")
  .prompt("Analyze this text: ...")
  .build();

Uni<ProviderResponse> result = provider.infer(request);

// 4. Monitor execution (Monitor Inference skill)
result.onItem().invoke(response -> {
  metricsRegistry.timer("inference.latency",
    "tenant", tenant.getTenantId()
  ).record(response.getLatencyMs(), TimeUnit.MILLISECONDS);
});
```

## Best Practices

1. **Chain Skills** - Use multiple skills in sequence for complete tasks
2. **Check Prerequisites** - Ensure each skill's requirements are met
3. **Handle Errors** - Each skill includes error handling guidance
4. **Monitor Execution** - Always use monitoring for production
5. **Isolate Tenants** - Always include tenant context for multi-tenant systems
6. **Test Locally** - Test skill instructions locally before production

## Troubleshooting

If a skill isn't working:

1. Check the SKILL.md for prerequisites
2. Review error handling section
3. Look at references/ documentation
4. Check inference-golek logs
5. Use `monitor-inference` skill to debug

## Contributing

To add new skills:

1. Create new skill directory
2. Write SKILL.md with instructions
3. Add scripts/ and references/
4. Test with agent
5. Submit as contribution

## Links

- [Inference-Golek Documentation](../README.md)
- [Agent Skills Standard](https://agentskills.io)
- [OpenAI Codex Skills](https://developers.openai.com/codex/skills/)
