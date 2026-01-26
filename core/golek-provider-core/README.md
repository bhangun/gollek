

```
inference-providers-spi/
├── pom.xml
└── src/main/java/tech/kayys/wayang/inference/providers/
    ├── core/
    │   ├── LLMProvider.java
    │   ├── ProviderCapabilities.java
    │   ├── ProviderRequest.java
    │   ├── ProviderResponse.java
    │   ├── ProviderContext.java
    │   ├── ProviderException.java
    │   └── ProviderMetrics.java
    ├── streaming/
    │   ├── StreamingProvider.java
    │   ├── StreamChunk.java
    │   └── StreamingResponse.java
    ├── circuit/
    │   ├── CircuitBreaker.java
    │   ├── CircuitBreakerConfig.java
    │   └── CircuitBreakerOpenException.java
    └── registry/
        ├── ProviderRegistry.java
        └── ProviderDescriptor.java
```



## 📋 Summary

This implementation provides:

### ✅ **Provider System**
- Clean SPI with `LLMProvider` interface
- `ProviderRegistry` for discovery and management
- Streaming support via `StreamingLLMProvider`
- Health checks and metrics
- Tenant-aware isolation

### ✅ **Observability**
- `InferenceObserver` for lifecycle hooks
- Metrics via Micrometer
- Distributed tracing via OpenTelemetry
- Structured logging

### ✅ **Safety**
- `SafetyPlugin` interface
- Content moderation implementation
- Configurable patterns
- Violation tracking



### ✅ **PyTorch Provider**
- `.pt`, `.pth`, TorchScript support
- CUDA acceleration
- Streaming generation
- Dynamic quantization

### ✅ **TensorFlow Provider**
- SavedModel format
- TensorRT optimization
- Batch inference
- Signature-based serving

### ✅ **HuggingFace Provider**
- Cloud API integration
- Serverless inference
- Streaming support
- Model Hub access

### ✅ **Embedding Provider**
- Sentence transformers
- Batch embedding
- Normalized vectors
- Multiple pooling strategies

### 🎯 **Key Features**
- Multi-tenant isolation
- Health checks
- Metrics integration
- Graceful degradation
- Model caching
- Format auto-detection