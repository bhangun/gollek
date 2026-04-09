# Gollek SDK :: ML Framework Façade

The `gollek-sdk-ml` module is the top-level aggregator for the Gollek ML framework. It provides the central `GollekSdk` façade, exposing user-friendly entry points for common ML tasks without requiring the developer to instantiate complex internal pipelines manually.

## Features

- **Model Builder API**: Simplifies the loading of models, tokenizers, and weights via `GollekSdk.builder()`.
- **Pre-configured Pipelines**: Provides native access to `TextGenerationPipeline`, `EmbeddingsPipeline`, etc.
- **Unified Interface**: Masks the complexity of `gollek-sdk-autograd`, `gollek-sdk-nn`, and the execution kernels.

## Example Usage

```java
import tech.kayys.gollek.ml.Gollek;

// Create an instance tied to the local execution backend
Gollek gollek = Gollek.builder()
    .model("Qwen/Qwen2.5-0.5B")
    .device("METAL")
    .build();

// Simple text completion hiding tokenizer complexity
String answer = gollek.createCompletion("What is the capital of France?");
```
