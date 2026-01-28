---
name: load-model-from-repository
description: Load ML models from repositories (Local, HuggingFace, cloud storage) with caching and hardware detection
metadata:
  short-description: Load models from multiple sources
  category: models
  difficulty: beginner
---

# Load Model from Repository Skill

Load and manage ML models from various repositories: local filesystem, HuggingFace, cloud storage.

## When to Use

- You need to load models from HuggingFace Hub
- You want to cache models locally
- You need hardware-aware model selection
- You want automatic model versioning

## Supported Repositories

1. **Local Repository** - File system-based models
2. **HuggingFace** - Direct from HuggingFace Hub
3. **Cloud Storage** - S3, GCS, Azure Blob

## Steps

### 1. Initialize Model Repository

```java
ModelRepository repository = ModelRepositoryRegistry
  .getRepository("huggingface");

Uni<Void> init = repository.initialize(Map.of(
  "cache_path", "/opt/models/cache",
  "cache_size_gb", "50",
  "hf_token", System.getenv("HF_TOKEN")
));
```

### 2. Query Available Models

```java
Uni<List<ModelInfo>> models = repository.listModels(
  "huggingface",
  Map.of("task", "text-classification")
);

models.onItem().invoke(list -> {
  list.forEach(model -> {
    System.out.println(model.getId() + " v" + model.getVersion());
  });
});
```

### 3. Load Model with Hardware Detection

```java
HardwareDetector detector = new HardwareDetector();
HardwareCapabilities hardware = detector.detect();

System.out.println("GPU Available: " + hardware.hasGPU());
System.out.println("RAM: " + hardware.getMemoryGB() + "GB");

ModelRef model = repository.load(
  "bert-base-uncased",
  Map.of(
    "device", hardware.hasGPU() ? "cuda" : "cpu",
    "quantize", hardware.getMemoryGB() < 4 ? "int8" : "float32"
  )
);
```

### 4. Access Loaded Model

```java
Model model = model.getModel();
ModelMetadata metadata = model.getMetadata();

System.out.println("Framework: " + metadata.getFramework());    // tensorflow, pytorch, onnx
System.out.println("Input Shape: " + metadata.getInputShape());
System.out.println("Output Shape: " + metadata.getOutputShape());
```

## Repository Types

### Local Repository

```java
repository = ModelRepositoryRegistry.getRepository("local");
repository.load("./models/my-model", Map.of());
```

### HuggingFace Repository

```java
// Automatically downloads and caches
repository = ModelRepositoryRegistry.getRepository("huggingface");
repository.load("bert-base-uncased", Map.of(
  "use_cache", "true"
));
```

### Cloud Storage (S3)

```java
repository = ModelRepositoryRegistry.getRepository("s3");
repository.load("s3://bucket/models/model-v1.onnx", Map.of(
  "region", "us-east-1"
));
```

## Model Caching

```java
// Cache automatically handles:
// - Disk space management
// - LRU eviction policy
// - Model versioning
// - Metadata caching

CachedModelRepository cached = new CachedModelRepository(
  underlying,
  Map.of(
    "max_cache_size_gb", "100",
    "eviction_policy", "lru"
  )
);
```

## Hardware Detection

```java
HardwareDetector detector = new HardwareDetector();
HardwareCapabilities caps = detector.detect();

// Available devices
caps.getAvailableDevices()     // [cpu, cuda, tensorrt]
caps.getMemoryGB()            // Total system RAM
caps.getGPUMemoryGB()          // GPU VRAM (if available)
caps.getSupportedPrecisions()  // [float32, float16, int8]
```

## Model Selection Policy

```java
SelectionPolicy policy = new SelectionPolicy()
  .preferPrecision("int8")              // Quantized for speed
  .preferDevice("cuda")                 // GPU if available
  .fallbackDevice("cpu");               // CPU as fallback

ModelRef selected = repository.selectModel(
  "bert-*",
  policy
);
```

## Error Handling

```java
repository.load("model-id")
  .onFailure().recover(ex -> {
    if (ex instanceof ModelNotFound) {
      // Try alternative model
      return repository.load("fallback-model");
    }
    throw new RuntimeException(ex);
  });
```

## Performance Considerations

- **First Load**: Slow (downloads/copies model)
- **Cached Load**: Fast (loads from disk cache)
- **Memory**: Keep models loaded in warm pool for quick reuse
- **Disk**: Monitor cache size, implement cleanup policies

## See Also

- [Run Inference](./run-inference.md)
- [Model Pool Management](../references/model-pool.md)
- [Hardware Detection](../references/hardware.md)
