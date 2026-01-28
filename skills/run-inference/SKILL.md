---
name: run-inference
description: Execute inference using inference-golek with support for multiple model formats (GGUF, ONNX, Triton, Cloud APIs)
metadata:
  short-description: Run ML model inference
  category: inference
  difficulty: beginner
---

# Run Inference Skill

Execute ML model inference using inference-golek. Supports GGUF (llama.cpp), ONNX Runtime, Triton Inference Server, and cloud APIs.

## When to Use

- You need to run inference on an ML model
- You want multi-format model support (GGUF, ONNX, Triton, OpenAI, etc.)
- You need circuit breaker and rate limiting
- You want tenant isolation and multi-tenancy support

## Prerequisites

1. inference-golek runtime configured
2. Model available in one of the supported formats
3. Provider initialized (OpenAI, Anthropic, local ONNX, etc.)

## Steps

### 1. Initialize Provider

```bash
# If using cloud provider (OpenAI, Anthropic, etc.)
export OPENAI_API_KEY=your-key
export PROVIDER_TYPE=openai

# If using local model (ONNX, GGUF)
export MODEL_PATH=/path/to/model
export PROVIDER_TYPE=onnx
```

### 2. Create Inference Request

```java
ProviderRequest request = ProviderRequest.builder()
  .tenantId("tenant-123")
  .userId("user-456")
  .modelId("bert-base-uncased")
  .prompt("Classify this text: ...")
  .parameters(Map.of(
    "maxTokens", 100,
    "temperature", 0.7
  ))
  .build();
```

### 3. Execute Inference

```java
Uni<ProviderResponse> result = provider.infer(request);

result.onItem().invoke(response -> {
  System.out.println("Output: " + response.getOutput());
  System.out.println("Latency: " + response.getLatencyMs() + "ms");
  System.out.println("Provider: " + response.getProvider());
});
```

### 4. Handle Response

```java
response.getOutput()           // Model output
response.getLatencyMs()        // Execution time
response.getTokensUsed()       // Token count (if applicable)
response.getMetadata()         // Provider-specific metadata
response.getUsageMetrics()     // Resource usage
```

## Options

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tenantId` | String | required | Multi-tenant isolation |
| `userId` | String | required | User context for audit |
| `modelId` | String | required | Model identifier |
| `prompt` | String | required | Input to model |
| `maxTokens` | Integer | 100 | Max output length |
| `temperature` | Float | 0.7 | Sampling temperature |
| `timeout` | Duration | 30s | Request timeout |

### Provider Types

- **openai** - OpenAI API (GPT-3.5, GPT-4)
- **anthropic** - Anthropic Claude
- **google** - Google PaLM / Gemini
- **onnx** - Local ONNX Runtime
- **gguf** - llama.cpp format
- **triton** - Triton Inference Server

## Example: Text Classification

```bash
# Request inference for sentiment analysis
curl -X POST http://localhost:8080/api/infer \
  -H "Content-Type: application/json" \
  -H "X-Tenant-Id: tenant-1" \
  -d '{
    "modelId": "distilbert-sentiment",
    "prompt": "This movie was amazing!",
    "parameters": {
      "maxTokens": 50
    }
  }'
```

## Example: Multi-Provider Fallback

```java
// Try primary provider, fallback to secondary
Uni<ProviderResponse> result = openaiProvider.infer(request)
  .onFailure().recoverWithUni(() -> 
    anthropicProvider.infer(request)
  );
```

## Error Handling

- **Rate Limited**: Request queued or rejected based on quota
- **Circuit Breaker Open**: Provider temporarily unavailable, fallback attempted
- **Timeout**: Request exceeded max duration
- **Invalid Input**: Validation error returned

## Tenant Isolation

Each request is isolated by tenant:

```java
request.setTenantId("tenant-123");  // Separate resource quotas
request.setUserId("user-456");      // Individual audit trail
```

## Performance Tips

1. **Reuse Provider Instance**: Initialize once, use multiple times
2. **Batch Requests**: Group multiple inferences when possible
3. **Use Appropriate Timeout**: Balance between responsiveness and completion
4. **Monitor Latency**: Track response times for optimization
5. **Cache Models**: Use warm model pools to avoid cold starts

## Troubleshooting

### Provider Not Initialized
```
Error: Provider not initialized
Fix: Call provider.initialize(config) before infer()
```

### Rate Limit Exceeded
```
Error: Rate limit exceeded
Fix: Implement backoff and retry logic
```

### Model Not Found
```
Error: Model not found: xyz
Fix: Verify model is available in the provider
```

## See Also

- [Load Model Skill](#load-model)
- [Model Repository Skill](#model-repository)
- [Workflow Integration](../references/workflow-integration.md)
