# Gollek SDK LiteRT Module

LiteRT (TensorFlow Lite) integration module for the Gollek SDK, providing high-level APIs for mobile and edge device inference.

## Features

- ✅ **Model Management**: Load, unload, and introspect LiteRT models
- ✅ **Inference APIs**: Synchronous and asynchronous inference
- ✅ **Batch Processing**: Efficient batch inference support
- ✅ **Performance Metrics**: Latency percentiles, memory tracking
- ✅ **Hardware Acceleration**: CPU, GPU, NNAPI, CoreML delegates
- ✅ **CLI Integration**: Full command-line interface for model management

## Installation

Add the dependency to your `pom.xml`:

```xml
<dependency>
    <groupId>tech.kayys.gollek</groupId>
    <artifactId>gollek-sdk-litert</artifactId>
    <version>${gollek.version}</version>
</dependency>
```

## Usage

### Basic Usage

```java
import tech.kayys.gollek.sdk.litertLiteRTSdk;
import tech.kayys.gollek.sdk.litertconfig.LiteRTConfig;

// Create SDK instance with default config
LiteRTSdk sdk = new LiteRTSdk();

// Load a model
sdk.loadModel("mobilenet", Path.of("/models/mobilenet.litertlm"));

// Run inference
InferenceRequest request = InferenceRequest.builder()
    .model("mobilenet")
    .inputData(inputTensor)
    .build();

InferenceResponse response = sdk.infer(request);

// Clean up
sdk.close();
```

### Advanced Configuration

```java
LiteRTConfig config = LiteRTConfig.builder()
    .numThreads(4)
    .delegate(LiteRTConfig.Delegate.GPU)
    .enableXnnpack(true)
    .useMemoryPool(true)
    .poolSizeBytes(32 * 1024 * 1024) // 32MB
    .cacheDir("/tmp/litert-cache")
    .build();

LiteRTSdk sdk = new LiteRTSdk(config);
```

### Async Inference

```java
CompletableFuture<InferenceResponse> future = sdk.inferAsync(request);

future.thenAccept(response -> {
    System.out.println("Inference completed");
    // Process response
}).exceptionally(error -> {
    System.err.println("Inference failed: " + error.getMessage());
    return null;
});
```

### Batch Inference

```java
List<InferenceRequest> requests = List.of(request1, request2, request3);

// Synchronous batch
List<InferenceResponse> responses = sdk.inferBatch(requests);

// Asynchronous batch
CompletableFuture<List<InferenceResponse>> future = sdk.inferBatchAsync(requests);
```

### Model Introspection

```java
// List loaded models
List<String> models = sdk.listModels();

// Get model information
LiteRTModelInfo info = sdk.getModelInfo("mobilenet");
System.out.println("Model: " + info.getModelId());
System.out.println("Size: " + info.getModelSizeBytes() + " bytes");
System.out.println("Inputs: " + info.getInputs().size());
System.out.println("Outputs: " + info.getOutputs().size());
```

### Performance Metrics

```java
// Get metrics
LiteRTMetrics metrics = sdk.getMetrics("mobilenet");
System.out.println("Total inferences: " + metrics.getTotalInferences());
System.out.println("Avg latency: " + metrics.getAvgLatencyMs() + " ms");
System.out.println("P95 latency: " + metrics.getP95LatencyMs() + " ms");
System.out.println("Peak memory: " + metrics.getPeakMemoryBytes() + " bytes");

// Reset metrics
sdk.resetMetrics();
```

## CLI Usage

The LiteRT module integrates with the Gollek CLI:

```bash
# List loaded models
gollek litert list

# Load a model
gollek litert load mobilenet /models/mobilenet.litertlm

# Load with custom configuration
gollek litert load mobilenet /models/mobilenet.litertlm \
  --threads 4 \
  --delegate GPU

# Show model information
gollek litert info mobilenet

# Run inference
gollek litert infer mobilenet input.bin --output output.bin

# Run inference asynchronously
gollek litert infer mobilenet input.bin --async

# View performance metrics
gollek litert metrics mobilenet

# Reset metrics
gollek litert metrics --reset

# Unload a model
gollek litert unload mobilenet
```

## Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `numThreads` | int | 4 | Number of CPU threads |
| `delegate` | Delegate | AUTO | Hardware delegate (NONE, CPU, GPU, NNAPI, COREML, AUTO) |
| `enableXnnpack` | boolean | true | Enable XNNPACK optimization |
| `useMemoryPool` | boolean | true | Use memory pool for faster allocations |
| `poolSizeBytes` | long | 0 (16MB) | Memory pool size in bytes |
| `cacheDir` | String | null | Model cache directory |

## Hardware Delegates

### CPU (XNNPACK)
- Optimized for ARM and x86 CPUs
- Uses SIMD instructions
- Best for: General inference on CPU

### GPU (Metal/OpenCL)
- iOS/macOS: Metal delegate
- Android: OpenCL delegate
- Best for: High-throughput inference

### NNAPI (Android only)
- Android Neural Networks API
- Automatically selects best accelerator (DSP, GPU, NPU)
- Best for: Android devices with hardware accelerators

### CoreML (iOS only)
- Apple Core ML framework
- Uses Apple Neural Engine (ANE)
- Best for: iOS devices with ANE

## Model Format

LiteRT models use the `.litertlm` file extension (formerly `.tflite`). Models should be in TensorFlow Lite FlatBuffer format.

### Converting Models

Convert your TensorFlow models to LiteRT format:

```python
import tensorflow as tf

# Convert to LiteRT format
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_dir')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
litert_model = converter.convert()

# Save with .litertlm extension
with open('model.litertlm', 'wb') as f:
    f.write(litert_model)
```

## Performance Tips

1. **Use Hardware Delegates**: GPU or NPU for better performance
2. **Enable XNNPACK**: Optimized CPU inference
3. **Use Memory Pool**: Reduces allocation overhead
4. **Batch Inference**: Process multiple inputs together
5. **Warm Up Models**: Run a dummy inference to initialize caches
6. **Monitor Metrics**: Track latency and memory usage

## Thread Safety

The `LiteRTSdk` class is thread-safe. Multiple threads can call `infer()` and `inferAsync()` concurrently. However, model loading and unloading should be done sequentially.

## Resource Management

Always close the SDK when done:

```java
try (LiteRTSdk sdk = new LiteRTSdk()) {
    // Use SDK
} // Automatically closes and frees resources
```

Or manually:

```java
LiteRTSdk sdk = new LiteRTSdk();
try {
    // Use SDK
} finally {
    sdk.close();
}
```

## Error Handling

```java
try {
    sdk.loadModel("model", Path.of("model.litertlm"));
    InferenceResponse response = sdk.infer(request);
} catch (LiteRTException e) {
    System.err.println("LiteRT error: " + e.getMessage());
    // Handle error
}
```

## Integration with Gollek Platform

The LiteRT SDK module integrates seamlessly with the broader Gollek platform:

- Works with Gollek's model registry
- Supports Gollek's authentication and authorization
- Integrates with Gollek's monitoring and observability stack
- Compatible with Gollek's workflow engine

## Examples

See the [examples directory](../../../examples/litert/) for complete working examples:

- Basic inference
- Image classification
- Object detection
- Text classification
- Batch processing
- Streaming inference

## API Reference

### LiteRTSdk

| Method | Description |
|--------|-------------|
| `loadModel(modelId, path)` | Load model from file |
| `loadModel(modelId, data)` | Load model from bytes |
| `infer(request)` | Synchronous inference |
| `inferAsync(request)` | Asynchronous inference |
| `inferBatch(requests)` | Batch inference |
| `inferBatchAsync(requests)` | Async batch inference |
| `getModelInfo(modelId)` | Get model information |
| `listModels()` | List loaded models |
| `unloadModel(modelId)` | Unload model |
| `getMetrics(modelId)` | Get performance metrics |
| `resetMetrics()` | Reset metrics |
| `isAvailable()` | Check LiteRT availability |
| `getVersion()` | Get LiteRT version |
| `close()` | Close and free resources |

## License

MIT License - Copyright (c) 2026 Kayys.tech
