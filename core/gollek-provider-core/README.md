# Gollek Provider Core (SPI)

This module defines the **provider SPI** and shared contracts used by all model providers (cloud and local).

## Key Capabilities

* Provider interfaces for sync and streaming inference
* Health, metrics, and capability reporting
* Provider registry + discovery hooks
* Common rate‑limit and audit helpers

## Core Interfaces (Current Paths)

* `inference-gollek/core/gollek-provider-core/src/main/java/tech/kayys/gollek/provider/core/spi/LLMProvider.java`
* `inference-gollek/core/gollek-provider-core/src/main/java/tech/kayys/gollek/provider/core/spi/StreamingProvider.java`
* `inference-gollek/core/gollek-provider-core/src/main/java/tech/kayys/gollek/provider/core/spi/ProviderContext.java`
* `inference-gollek/core/gollek-provider-core/src/main/java/tech/kayys/gollek/provider/core/spi/ProviderCandidate.java`

## Streaming Helpers

* `inference-gollek/core/gollek-provider-core/src/main/java/tech/kayys/gollek/provider/core/streaming/StreamHandler.java`
* `inference-gollek/core/gollek-provider-core/src/main/java/tech/kayys/gollek/provider/core/streaming/SSEStreamHandler.java`
* `inference-gollek/core/gollek-provider-core/src/main/java/tech/kayys/gollek/provider/core/streaming/WebSocketStreamHandler.java`

## Observability & Reliability

* `inference-gollek/core/gollek-provider-core/src/main/java/tech/kayys/gollek/provider/core/ratelimit/RateLimiter.java`
* `inference-gollek/core/gollek-provider-core/src/main/java/tech/kayys/gollek/provider/core/quota/ProviderQuotaService.java`
* `inference-gollek/core/gollek-provider-core/src/main/java/tech/kayys/gollek/provider/core/audit/AuditLoggingPlugin.java`

## Notes

* Cloud/local providers live under `inference-gollek/provider/`
* Routing policies are implemented in `gollek-engine` (see `ModelRouterService`)

#### [NEW] Strategy Implementations

| Strategy | File |
|----------|------|
| `RoundRobinStrategy` | `strategy/RoundRobinStrategy.java` |
| `WeightedRandomStrategy` | `strategy/WeightedRandomStrategy.java` |
| `LeastLoadedStrategy` | `strategy/LeastLoadedStrategy.java` |
| `CostOptimizedStrategy` | `strategy/CostOptimizedStrategy.java` |
| `LatencyOptimizedStrategy` | `strategy/LatencyOptimizedStrategy.java` |
| `FailoverStrategy` | `strategy/FailoverStrategy.java` |

---

### 4. Quota Integration

Add quota checking before inference:
```java
protected Uni<InferenceResponse> infer(...) {
    return checkQuota(requestId, providerId)
        .onItem().transformToUni(allowed -> {
            if (!allowed) {
                throw new QuotaExhaustedException(providerId);
            }
            return doInfer(request);
        });
}
```

Catchable exception to trigger failover routing.

---

### 5. Configuration Schema

#### Example `application.yaml`:

```yaml
gollek:
  routing:
    default-strategy: FAILOVER
    auto-failover: true
    max-retries: 3
    
    pools:
      - id: cloud-primary
        type: CLOUD
        providers: [gemini, openai, anthropic]
        strategy: WEIGHTED_RANDOM
        weights:
          gemini: 50
          openai: 30
          anthropic: 20
          
      - id: local-production
        type: LOCAL
        providers: [local-vllm, ollama]
        strategy: LEAST_LOADED
        
      - id: local-dev
        type: LOCAL
        providers: [ollama, local]
        strategy: ROUND_ROBIN
```

---

## Verification Plan

### Automated Tests

1. **Unit tests** for each `SelectionStrategy`
2. **Integration tests** for `MultiProviderRouter` with mock providers
3. **Quota exhaustion test**: verify failover triggers correctly

### Manual Verification

1. Deploy with multiple providers configured
2. Test round-robin cycles through providers
3. Exhaust quota on one provider, verify auto-switch
4. Test user-selected routing via API parameter



# Multi-Provider Routing System Walkthrough

## Summary

Implemented a comprehensive multi-provider routing system for the Gollek inference server, enabling load balancing, automatic failover, and configurable selection strategies across cloud and local providers.

---

## Changes Made

### Phase 1: API Types (`gollek-spi/routing`)

| File | Description |
|------|-------------|
| [SelectionStrategy.java](inference-gollek/core/gollek-spi/src/main/java/tech/kayys/gollek/api/routing/SelectionStrategy.java) | Enum with 9 selection strategies |
| [ProviderPool.java](inference-gollek/core/gollek-spi/src/main/java/tech/kayys/gollek/api/routing/ProviderPool.java) | Record for grouping providers by type |
| [RoutingConfig.java](inference-gollek/core/gollek-spi/src/main/java/tech/kayys/gollek/api/routing/RoutingConfig.java) | Configuration for pools, weights, failover |
| [RoutingDecision.java](inference-gollek/core/gollek-spi/src/main/java/tech/kayys/gollek/api/routing/RoutingDecision.java) | Result of routing with fallback info |
| [QuotaExhaustedException.java](inference-gollek/core/gollek-spi/src/main/java/tech/kayys/gollek/api/routing/QuotaExhaustedException.java) | Exception triggering failover |

### Enhanced Existing

| File | Changes |
|------|---------|
| [RoutingContext.java](inference-gollek/core/gollek-spi/src/main/java/tech/kayys/gollek/api/provider/RoutingContext.java) | Added `strategyOverride`, [poolId](inference-gollek/core/gollek-spi/src/main/java/tech/kayys/gollek/api/routing/ProviderPool.java#114-118), [excludedProviders](inference-gollek/core/gollek-spi/src/main/java/tech/kayys/gollek/api/provider/RoutingContext.java#169-173) |

---

### Phase 2: Core Router (`gollek-provider-core/routing`)

| File | Description |
|------|-------------|
| [MultiProviderRouter.java](inference-gollek/core/gollek-provider-core/src/main/java/tech/kayys/gollek/provider/core/routing/MultiProviderRouter.java) | Central router with failover support |
| [ProviderSelector.java](inference-gollek/core/gollek-provider-core/src/main/java/tech/kayys/gollek/provider/core/routing/strategy/ProviderSelector.java) | Strategy interface |
| [ProviderSelection.java](inference-gollek/core/gollek-provider-core/src/main/java/tech/kayys/gollek/provider/core/routing/strategy/ProviderSelection.java) | Selection result record |

### Selection Strategies

| Strategy | File | Description |
|----------|------|-------------|
| ROUND_ROBIN | [RoundRobinSelector.java](inference-gollek/core/gollek-provider-core/src/main/java/tech/kayys/gollek/provider/core/routing/strategy/RoundRobinSelector.java) | Cycles sequentially |
| RANDOM | [RandomSelector.java](inference-gollek/core/gollek-provider-core/src/main/java/tech/kayys/gollek/provider/core/routing/strategy/RandomSelector.java) | Equal probability |
| WEIGHTED_RANDOM | [WeightedRandomSelector.java](inference-gollek/core/gollek-provider-core/src/main/java/tech/kayys/gollek/provider/core/routing/strategy/WeightedRandomSelector.java) | Configurable weights |
| LEAST_LOADED | [LeastLoadedSelector.java](inference-gollek/core/gollek-provider-core/src/main/java/tech/kayys/gollek/provider/core/routing/strategy/LeastLoadedSelector.java) | Tracks active requests |
| COST_OPTIMIZED | [CostOptimizedSelector.java](inference-gollek/core/gollek-provider-core/src/main/java/tech/kayys/gollek/provider/core/routing/strategy/CostOptimizedSelector.java) | Prefers local providers |
| LATENCY_OPTIMIZED | [LatencyOptimizedSelector.java](inference-gollek/core/gollek-provider-core/src/main/java/tech/kayys/gollek/provider/core/routing/strategy/LatencyOptimizedSelector.java) | P95 latency tracking |
| FAILOVER | [FailoverSelector.java](inference-gollek/core/gollek-provider-core/src/main/java/tech/kayys/gollek/provider/core/routing/strategy/FailoverSelector.java) | Primary with fallback chain |
| SCORED | [ScoredSelector.java](inference-gollek/core/gollek-provider-core/src/main/java/tech/kayys/gollek/provider/core/routing/strategy/ScoredSelector.java) | Multi-factor scoring |
| USER_SELECTED | [UserSelectedSelector.java](inference-gollek/core/gollek-provider-core/src/main/java/tech/kayys/gollek/provider/core/routing/strategy/UserSelectedSelector.java) | Strict user preference |

---

### Phase 3: Model-Provider Mapping

| File | Description |
|------|-------------|
| [ModelProviderMapping.java](inference-gollek/core/gollek-spi/src/main/java/tech/kayys/gollek/api/routing/ModelProviderMapping.java) | Record mapping models to providers |
| [ModelProviderRegistry.java](inference-gollek/core/gollek-provider-core/src/main/java/tech/kayys/gollek/provider/core/routing/ModelProviderRegistry.java) | Registry with default mappings |

**Default Model Mappings:**

| Model | Providers |
|-------|----------|
| `gpt-4`, `gpt-4-turbo`, `gpt-3.5-turbo` | `openai`, `azure-openai` |
| `claude-3-opus`, `claude-3-sonnet` | `anthropic` |
| `gemini-pro`, `gemini-ultra` | `gemini` |
| `llama-3-8b`, `mistral-7b`, `phi-3` | `ollama`, [local](inference-gollek/core/gollek-spi/src/main/java/tech/kayys/gollek/api/routing/ProviderPool.java#85-98), `local-vllm` |
| `llama-3-70b` | `local-vllm` (large model) |
| `codellama-13b` | `ollama`, [local](inference-gollek/core/gollek-spi/src/main/java/tech/kayys/gollek/api/routing/ProviderPool.java#85-98) |

---

### Phase 4: Provider Implementations

Implemented adapters for **Ollama** (local), **Gemini** (cloud), and **OpenAI** (cloud), all extending the standard provider interfaces with:
- **Streaming Support**: Unified `Multi<StreamChunk>` API
- **Health Checks**: Standardized health probes returning status and latency
- **Configuration**: Quarkus Config mappings (e.g., `gollek.provider.openai.api-key`)

| Provider | Key Classes | Features |
|----------|-------------|----------|
| **Ollama** | [OllamaProvider](inference-gollek/provider/gollek-ext-cloud-ollama/src/main/java/tech/kayys/gollek/provider/ollama/OllamaProvider.java#32-234), [OllamaClient](inference-gollek/provider/gollek-ext-cloud-ollama/src/main/java/tech/kayys/gollek/provider/ollama/OllamaClient.java#12-69), [OllamaConfig](inference-gollek/provider/gollek-ext-cloud-ollama/src/main/java/tech/kayys/gollek/provider/ollama/OllamaConfig.java#13-64) | Local inference, embeddings, keep-alive control |
| **Gemini** | [GeminiProvider](inference-gollek/provider/gollek-ext-cloud-gemini/src/main/java/tech/kayys/gollek/provider/gemini/GeminiProvider.java#31-259), [GeminiClient](inference-gollek/provider/gollek-ext-cloud-gemini/src/main/java/tech/kayys/gollek/provider/gemini/GeminiClient.java#12-82), [GeminiConfig](inference-gollek/provider/gollek-ext-cloud-gemini/src/main/java/tech/kayys/gollek/provider/gemini/GeminiConfig.java#12-56) | 1M+ context, function calling, multimodal, safety settings |
| **OpenAI** | [OpenAIProvider](inference-gollek/provider/gollek-provider-openai/src/main/java/tech/kayys/gollek/provider/openai/OpenAIProvider.java#31-238), [OpenAIClient](inference-gollek/provider/gollek-provider-openai/src/main/java/tech/kayys/gollek/provider/openai/OpenAIClient.java#12-58), [OpenAIConfig](inference-gollek/provider/gollek-provider-openai/src/main/java/tech/kayys/gollek/provider/openai/OpenAIConfig.java#12-55) | GPT-4 Turbo, tool calling, structured outputs, embeddings |
| **Embedding** | [EmbeddingProvider](inference-gollek/provider/gollek-provider-embedding/src/main/java/tech/kayys/gollek/EmbeddingProvider.java#23-271), [EmbeddingConfig](inference-gollek/provider/gollek-provider-embedding/src/main/java/tech/kayys/gollek/provider/embedding/EmbeddingConfig.java#9-40) | Local embeddings (Sentence Transformers), mock generation |
| **Cerebras** | [CerebrasProvider](inference-gollek/provider/gollek-ext-cloud-cerebras/src/main/java/tech/kayys/gollek/provider/cerebras/CerebrasProvider.java#31-206), [CerebrasClient](inference-gollek/provider/gollek-ext-cloud-cerebras/src/main/java/tech/kayys/gollek/provider/cerebras/CerebrasClient.java#12-38), [CerebrasConfig](inference-gollek/provider/gollek-ext-cloud-cerebras/src/main/java/tech/kayys/gollek/provider/cerebras/CerebrasConfig.java#12-56) | Extreme speed Llama 3 inference (wafer-scale engine) |

---

## Usage Example

```java
// Configure router
RoutingConfig config = RoutingConfig.builder()
    .defaultStrategy(SelectionStrategy.FAILOVER)
    .pools(List.of(
        ProviderPool.cloudPool("cloud", List.of("gemini", "openai")),
        ProviderPool.localPool("local", List.of("ollama", "local-vllm"))
    ))
    .autoFailover(true)
    .build();

router.configure(config);

// Route request
RoutingContext ctx = RoutingContext.builder()
    .request(request)
    .requestContext(tenant)
    .costSensitive(true)  // Prefer local
    .build();

Uni<LLMProvider> provider = router.selectWithFailover(modelId, ctx);
```

---

## Next Steps

- [ ] Add quota integration to routing flow
- [ ] Add configuration file parsing (application.yaml)
- [ ] Write unit tests for selection strategies
