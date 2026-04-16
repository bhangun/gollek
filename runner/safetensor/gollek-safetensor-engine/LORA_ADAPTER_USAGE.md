# LoRA Adapter Usage Guide

## Overview

The `LoraAdapter` class provides comprehensive support for loading and managing LoRA (Low-Rank Adaptation) adapters in the Gollek SafeTensor inference engine. LoRA adapters enable efficient multi-tenant inference by allowing different fine-tuned versions of the same base model without maintaining multiple full model copies.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   LoraAdapterRouter                         │
│  (resolves adapter_id from request parameters)              │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                     LoraAdapter                             │
│  • Loads adapter_model.safetensors                          │
│  • Parses adapter_config.json                               │
│  • Caches loaded adapters                                   │
│  • Provides LoRA weight matrices (A, B) per module          │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              DirectInferenceEngine                          │
│  • Applies LoRA deltas during forward pass                  │
│  • W' = W + α × (B × A)                                     │
└─────────────────────────────────────────────────────────────┘
```

## Adapter Directory Structure

```
my-adapter/
├── adapter_model.safetensors    # LoRA weights (lora_A, lora_B matrices)
├── adapter_config.json          # Adapter metadata (rank, alpha, target modules)
└── README.md                    # Optional documentation
```

### adapter_config.json Format

```json
{
  "r": 16,                       # LoRA rank (dimension of low-rank decomposition)
  "lora_alpha": 32,              # Alpha scaling factor
  "lora_dropout": 0.0,           # Dropout rate (0.0 for inference)
  "bias": "none",                # Bias configuration
  "target_modules": [            # Modules to apply LoRA to
    "q_proj",
    "k_proj", 
    "v_proj",
    "o_proj"
  ],
  "task_type": "CAUSAL_LM",      # Task type
  "inference_mode": true,        # Exported for inference-only
  "base_model_name_or_path": "meta-llama/Llama-3-8B"
}
```

## Usage Examples

### 1. Basic Adapter Loading

```java
@Inject
LoraAdapter loraAdapter;

// Load an adapter from disk
Path adapterPath = Path.of("/models/adapters/sql-lora");
LoraAdapter.LoadedAdapter adapter = loraAdapter.load(adapterPath);

// Access adapter metadata
System.out.println("Rank: " + adapter.rank());           // e.g., 16
System.out.println("Alpha: " + adapter.alpha());         // e.g., 32.0
System.out.println("Scaling: " + adapter.scalingFactor()); // α/r = 2.0

// Cleanup when done
adapter.close();
```

### 2. Accessing LoRA Weights

```java
// Get LoRA matrices for a specific module
Optional<LoraAdapter.LoraPair> pair = 
    adapter.getLoraPair("model.layers.0.self_attn.q_proj");

if (pair.isPresent()) {
    Tensor loraA = pair.get().a();  // Down-projection (r × d)
    Tensor loraB = pair.get().b();  // Up-projection (k × r)
    
    // Apply LoRA: W' = W + scaling × (B × A)
    // (Actual application happens in DirectInferenceEngine)
}

// List all modules with LoRA weights
Set<String> modules = adapter.getModuleNames();
for (String module : modules) {
    System.out.println("Module: " + module);
}
```

### 3. Using LoraAdapterRouter (Request-Based Resolution)

```java
@Inject
LoraAdapterRouter router;

// Create a provider request with adapter parameters
Map<String, Object> parameters = Map.of(
    "adapter_id", "sql-lora",
    "prompt", "SELECT * FROM users WHERE"
);

ProviderRequest request = new ProviderRequest(
    "request-123",
    parameters,
    Map.of(),  // metadata
    null       // context
);

// Resolve and load adapter for this request
Optional<LoraAdapter.LoadedAdapter> adapter = router.resolve(request);

if (adapter.isPresent()) {
    // Use adapter for inference
    LoadedModel model = engine.getLoadedModel("llama-3-8b");
    // ... perform inference with LoRA applied
}
```

### 4. Adapter Registry (Pre-Registered Adapters)

```java
@Inject
LoraAdapterRouter router;

// Register adapters at startup
router.register("sql-lora", Path.of("/models/adapters/sql-lora"));
router.register("code-lora", Path.of("/models/adapters/code-lora"));
router.register("medical-lora", Path.of("/models/adapters/medical-lora"));

// List all registered adapters
Map<String, Path> adapters = router.registeredAdapters();
adapters.forEach((name, path) -> 
    System.out.println(name + " → " + path)
);

// Unregister an adapter
router.unregister("code-lora");
```

### 5. Configuration via application.properties

```properties
# Pre-register adapters at startup
gollek.safetensor.adapters.sql-lora=/models/adapters/sql-lora
gollek.safetensor.adapters.code-lora=/models/adapters/code-lora
gollek.safetensor.adapters.medical-lora=/models/adapters/medical-lora

# Adapter cache settings
gollek.safetensor.lora.cache-enabled=true
gollek.safetensor.lora.max-cached-adapters=50
```

### 6. Request Parameters

Clients can specify adapters via multiple parameter keys (first non-null wins):

```java
// Via adapter_id (uses registry)
Map.of("adapter_id", "sql-lora")

// Via direct path
Map.of("adapter_path", "/models/adapters/custom-lora")

// Aliases for compatibility
Map.of("lora_adapter_id", "sql-lora")
Map.of("lora_adapter_path", "/models/adapters/custom-lora")
Map.of("lora_adapter", "sql-lora")

// Via metadata
Map.of("metadata.adapter_id", "sql-lora")
```

### 7. Cache Management

```java
@Inject
LoraAdapter loraAdapter;

// Check if adapter is cached
boolean cached = loraAdapter.isLoaded(Path.of("/models/adapters/sql-lora"));

// Get cached adapter without reloading
Optional<LoraAdapter.LoadedAdapter> cachedAdapter = 
    loraAdapter.getCached(Path.of("/models/adapters/sql-lora"));

// Unload specific adapter
loraAdapter.unload(Path.of("/models/adapters/sql-lora"));

// Clear all cached adapters
loraAdapter.clearCache();

// List all cached adapter paths
Set<Path> cachedPaths = loraAdapter.getCachedAdapterPaths();
```

## LoRA Mathematics

The LoRA forward pass applies a low-rank decomposition to the base model weights:

```
W' = W + ΔW
ΔW = (α / r) × (B × A)

where:
  W  = base model weight matrix (k × d)
  A  = down-projection matrix (r × d)
  B  = up-projection matrix (k × r)
  r  = rank (typically 8-64)
  α  = alpha scaling factor
```

### Example Calculation

For a 4096×4096 projection with rank-16 LoRA:

```
Base weight W:     4096 × 4096 = 16,777,216 parameters (64 MB @ FP16)
LoRA A matrix:     16 × 4096   = 65,536 parameters (256 KB @ FP16)
LoRA B matrix:     4096 × 16   = 65,536 parameters (256 KB @ FP16)
Total adapter:     ~512 KB (vs 64 MB base model)

Scaling factor: α/r = 32/16 = 2.0
Forward pass: W' = W + 2.0 × (B × A)
```

## Performance Considerations

### Memory Efficiency

- **Base model**: Loaded once, memory-mapped via SafeTensors
- **LoRA adapters**: ~50 MB each (rank-16, LLaMA-3-8B)
- **Multi-tenant**: N tenants require 1 base + N adapters (not N full models)

### Caching Strategy

- Adapters are cached on first load
- Cache is keyed by absolute path
- Automatic eviction not implemented (use `clearCache()` manually)
- Thread-safe for concurrent access

### Threading

- `LoraAdapter` is thread-safe and can be injected as `@ApplicationScoped`
- `LoadedAdapter` is immutable after creation
- Tensor weights are read-only during inference
- Multiple requests can share the same cached adapter

## Error Handling

```java
try {
    LoraAdapter.LoadedAdapter adapter = loraAdapter.load(adapterPath);
} catch (IllegalArgumentException e) {
    // Invalid adapter path or missing files
    log.error("Adapter not found", e);
} catch (RuntimeException e) {
    // Loading failed (corrupt file, incompatible format, etc.)
    log.error("Failed to load adapter", e);
}
```

## Integration with DirectInferenceEngine

```java
@Inject
DirectInferenceEngine engine;

@Inject
LoraAdapterRouter router;

public String generateWithLora(ProviderRequest request) {
    // Load base model
    String modelKey = engine.loadModel(Path.of("/models/llama-3-8b"));
    LoadedModel model = engine.getLoadedModel(modelKey);
    
    // Resolve LoRA adapter
    Optional<LoraAdapter.LoadedAdapter> adapter = router.resolve(request);
    
    if (adapter.isPresent()) {
        // Adapter will be applied during forward pass
        // W' = W + α × (B × A) for each targeted layer
        return engine.generate(model, request, adapter.get());
    } else {
        // Base model only
        return engine.generate(model, request);
    }
}
```

## Troubleshooting

### Adapter Not Found

```
LoraAdapterRouter: adapter_id 'sql-lora' not found in registry
```

**Solution**: Register the adapter via config or runtime:
```java
router.register("sql-lora", Path.of("/models/adapters/sql-lora"));
```

### Rank Mismatch

```
LoraAdapter: LoRA A tensor '...' has rank 32, expected 16
```

**Solution**: Ensure adapter_config.json `r` value matches actual tensor dimensions.

### Missing Target Module

```
LoraAdapter: no LoRA weights found for module 'model.layers.0.mlp.gate_proj'
```

**Solution**: This is normal if the adapter doesn't target all modules. Check `target_modules` in config.

## Best Practices

1. **Pre-register adapters** at application startup for known tenants
2. **Use adapter_id** instead of direct paths for better security
3. **Validate adapter compatibility** with base model before deployment
4. **Monitor cache size** in production (use `getCachedAdapterPaths().size()`)
5. **Implement adapter versioning** in your deployment strategy
6. **Test adapter loading** as part of CI/CD pipeline

## See Also

- `LoraAdapterRouter` - Request-based adapter resolution
- `DirectInferenceEngine` - Base model loading and inference
- [PEFT LoRA Documentation](https://huggingface.co/docs/peft/en/conceptual_guides/lora)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
