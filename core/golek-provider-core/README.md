# Golek Provider Core (SPI)

This module defines the **provider SPI** and shared contracts used by all model providers (cloud and local).

## Key Capabilities

* Provider interfaces for sync and streaming inference
* Health, metrics, and capability reporting
* Provider registry + discovery hooks
* Common rate‑limit and audit helpers

## Core Interfaces (Current Paths)

* `inference-golek/core/golek-provider-core/src/main/java/tech/kayys/golek/provider/core/spi/LLMProvider.java`
* `inference-golek/core/golek-provider-core/src/main/java/tech/kayys/golek/provider/core/spi/StreamingProvider.java`
* `inference-golek/core/golek-provider-core/src/main/java/tech/kayys/golek/provider/core/spi/ProviderContext.java`
* `inference-golek/core/golek-provider-core/src/main/java/tech/kayys/golek/provider/core/spi/ProviderCandidate.java`

## Streaming Helpers

* `inference-golek/core/golek-provider-core/src/main/java/tech/kayys/golek/provider/core/streaming/StreamHandler.java`
* `inference-golek/core/golek-provider-core/src/main/java/tech/kayys/golek/provider/core/streaming/SSEStreamHandler.java`
* `inference-golek/core/golek-provider-core/src/main/java/tech/kayys/golek/provider/core/streaming/WebSocketStreamHandler.java`

## Observability & Reliability

* `inference-golek/core/golek-provider-core/src/main/java/tech/kayys/golek/provider/core/ratelimit/RateLimiter.java`
* `inference-golek/core/golek-provider-core/src/main/java/tech/kayys/golek/provider/core/quota/ProviderQuotaService.java`
* `inference-golek/core/golek-provider-core/src/main/java/tech/kayys/golek/provider/core/audit/AuditLoggingPlugin.java`

## Notes

* Cloud/local providers live under `inference-golek/provider/`
* Routing policies are implemented in `golek-engine` (see `ModelRouterService`)

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
golek:
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

Implemented a comprehensive multi-provider routing system for the Golek inference server, enabling load balancing, automatic failover, and configurable selection strategies across cloud and local providers.

---

## Changes Made

### Phase 1: API Types (`golek-spi/routing`)

| File | Description |
|------|-------------|
| [SelectionStrategy.java](inference-golek/core/golek-spi/src/main/java/tech/kayys/golek/api/routing/SelectionStrategy.java) | Enum with 9 selection strategies |
| [ProviderPool.java](inference-golek/core/golek-spi/src/main/java/tech/kayys/golek/api/routing/ProviderPool.java) | Record for grouping providers by type |
| [RoutingConfig.java](inference-golek/core/golek-spi/src/main/java/tech/kayys/golek/api/routing/RoutingConfig.java) | Configuration for pools, weights, failover |
| [RoutingDecision.java](inference-golek/core/golek-spi/src/main/java/tech/kayys/golek/api/routing/RoutingDecision.java) | Result of routing with fallback info |
| [QuotaExhaustedException.java](inference-golek/core/golek-spi/src/main/java/tech/kayys/golek/api/routing/QuotaExhaustedException.java) | Exception triggering failover |

### Enhanced Existing

| File | Changes |
|------|---------|
| [RoutingContext.java](inference-golek/core/golek-spi/src/main/java/tech/kayys/golek/api/provider/RoutingContext.java) | Added `strategyOverride`, [poolId](inference-golek/core/golek-spi/src/main/java/tech/kayys/golek/api/routing/ProviderPool.java#114-118), [excludedProviders](inference-golek/core/golek-spi/src/main/java/tech/kayys/golek/api/provider/RoutingContext.java#169-173) |

---

### Phase 2: Core Router (`golek-provider-core/routing`)

| File | Description |
|------|-------------|
| [MultiProviderRouter.java](inference-golek/core/golek-provider-core/src/main/java/tech/kayys/golek/provider/core/routing/MultiProviderRouter.java) | Central router with failover support |
| [ProviderSelector.java](inference-golek/core/golek-provider-core/src/main/java/tech/kayys/golek/provider/core/routing/strategy/ProviderSelector.java) | Strategy interface |
| [ProviderSelection.java](inference-golek/core/golek-provider-core/src/main/java/tech/kayys/golek/provider/core/routing/strategy/ProviderSelection.java) | Selection result record |

### Selection Strategies

| Strategy | File | Description |
|----------|------|-------------|
| ROUND_ROBIN | [RoundRobinSelector.java](inference-golek/core/golek-provider-core/src/main/java/tech/kayys/golek/provider/core/routing/strategy/RoundRobinSelector.java) | Cycles sequentially |
| RANDOM | [RandomSelector.java](inference-golek/core/golek-provider-core/src/main/java/tech/kayys/golek/provider/core/routing/strategy/RandomSelector.java) | Equal probability |
| WEIGHTED_RANDOM | [WeightedRandomSelector.java](inference-golek/core/golek-provider-core/src/main/java/tech/kayys/golek/provider/core/routing/strategy/WeightedRandomSelector.java) | Configurable weights |
| LEAST_LOADED | [LeastLoadedSelector.java](inference-golek/core/golek-provider-core/src/main/java/tech/kayys/golek/provider/core/routing/strategy/LeastLoadedSelector.java) | Tracks active requests |
| COST_OPTIMIZED | [CostOptimizedSelector.java](inference-golek/core/golek-provider-core/src/main/java/tech/kayys/golek/provider/core/routing/strategy/CostOptimizedSelector.java) | Prefers local providers |
| LATENCY_OPTIMIZED | [LatencyOptimizedSelector.java](inference-golek/core/golek-provider-core/src/main/java/tech/kayys/golek/provider/core/routing/strategy/LatencyOptimizedSelector.java) | P95 latency tracking |
| FAILOVER | [FailoverSelector.java](inference-golek/core/golek-provider-core/src/main/java/tech/kayys/golek/provider/core/routing/strategy/FailoverSelector.java) | Primary with fallback chain |
| SCORED | [ScoredSelector.java](inference-golek/core/golek-provider-core/src/main/java/tech/kayys/golek/provider/core/routing/strategy/ScoredSelector.java) | Multi-factor scoring |
| USER_SELECTED | [UserSelectedSelector.java](inference-golek/core/golek-provider-core/src/main/java/tech/kayys/golek/provider/core/routing/strategy/UserSelectedSelector.java) | Strict user preference |

---

### Phase 3: Model-Provider Mapping

| File | Description |
|------|-------------|
| [ModelProviderMapping.java](inference-golek/core/golek-spi/src/main/java/tech/kayys/golek/api/routing/ModelProviderMapping.java) | Record mapping models to providers |
| [ModelProviderRegistry.java](inference-golek/core/golek-provider-core/src/main/java/tech/kayys/golek/provider/core/routing/ModelProviderRegistry.java) | Registry with default mappings |

**Default Model Mappings:**

| Model | Providers |
|-------|----------|
| `gpt-4`, `gpt-4-turbo`, `gpt-3.5-turbo` | `openai`, `azure-openai` |
| `claude-3-opus`, `claude-3-sonnet` | `anthropic` |
| `gemini-pro`, `gemini-ultra` | `gemini` |
| `llama-3-8b`, `mistral-7b`, `phi-3` | `ollama`, [local](inference-golek/core/golek-spi/src/main/java/tech/kayys/golek/api/routing/ProviderPool.java#85-98), `local-vllm` |
| `llama-3-70b` | `local-vllm` (large model) |
| `codellama-13b` | `ollama`, [local](inference-golek/core/golek-spi/src/main/java/tech/kayys/golek/api/routing/ProviderPool.java#85-98) |

---

### Phase 4: Provider Implementations

Implemented adapters for **Ollama** (local), **Gemini** (cloud), and **OpenAI** (cloud), all extending the standard provider interfaces with:
- **Streaming Support**: Unified `Multi<StreamChunk>` API
- **Health Checks**: Standardized health probes returning status and latency
- **Configuration**: Quarkus Config mappings (e.g., `golek.provider.openai.api-key`)

| Provider | Key Classes | Features |
|----------|-------------|----------|
| **Ollama** | [OllamaProvider](inference-golek/provider/golek-ext-cloud-ollama/src/main/java/tech/kayys/golek/provider/ollama/OllamaProvider.java#32-234), [OllamaClient](inference-golek/provider/golek-ext-cloud-ollama/src/main/java/tech/kayys/golek/provider/ollama/OllamaClient.java#12-69), [OllamaConfig](inference-golek/provider/golek-ext-cloud-ollama/src/main/java/tech/kayys/golek/provider/ollama/OllamaConfig.java#13-64) | Local inference, embeddings, keep-alive control |
| **Gemini** | [GeminiProvider](inference-golek/provider/golek-ext-cloud-gemini/src/main/java/tech/kayys/golek/provider/gemini/GeminiProvider.java#31-259), [GeminiClient](inference-golek/provider/golek-ext-cloud-gemini/src/main/java/tech/kayys/golek/provider/gemini/GeminiClient.java#12-82), [GeminiConfig](inference-golek/provider/golek-ext-cloud-gemini/src/main/java/tech/kayys/golek/provider/gemini/GeminiConfig.java#12-56) | 1M+ context, function calling, multimodal, safety settings |
| **OpenAI** | [OpenAIProvider](inference-golek/provider/golek-provider-openai/src/main/java/tech/kayys/golek/provider/openai/OpenAIProvider.java#31-238), [OpenAIClient](inference-golek/provider/golek-provider-openai/src/main/java/tech/kayys/golek/provider/openai/OpenAIClient.java#12-58), [OpenAIConfig](inference-golek/provider/golek-provider-openai/src/main/java/tech/kayys/golek/provider/openai/OpenAIConfig.java#12-55) | GPT-4 Turbo, tool calling, structured outputs, embeddings |
| **Embedding** | [EmbeddingProvider](inference-golek/provider/golek-provider-embedding/src/main/java/tech/kayys/golek/EmbeddingProvider.java#23-271), [EmbeddingConfig](inference-golek/provider/golek-provider-embedding/src/main/java/tech/kayys/golek/provider/embedding/EmbeddingConfig.java#9-40) | Local embeddings (Sentence Transformers), mock generation |
| **Cerebras** | [CerebrasProvider](inference-golek/provider/golek-ext-cloud-cerebras/src/main/java/tech/kayys/golek/provider/cerebras/CerebrasProvider.java#31-206), [CerebrasClient](inference-golek/provider/golek-ext-cloud-cerebras/src/main/java/tech/kayys/golek/provider/cerebras/CerebrasClient.java#12-38), [CerebrasConfig](inference-golek/provider/golek-ext-cloud-cerebras/src/main/java/tech/kayys/golek/provider/cerebras/CerebrasConfig.java#12-56) | Extreme speed Llama 3 inference (wafer-scale engine) |

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
