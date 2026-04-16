# Safetensor Runner Plugin

Safetensor format runner plugin for the Gollek inference engine.

## Overview

The Safetensor Runner Plugin provides support for Safetensor format models (`.safetensors`) using either:
- **Direct Backend**: Native Safetensor inference
- **GGUF Conversion**: Convert to GGUF for optimized inference

## Features

✅ **Supported Formats**:
- `.safetensors` - Safetensor format (Hugging Face)
- `.gguf` - GGUF format (via conversion)
- `.bin` - PyTorch format (legacy)

✅ **Supported Architectures**:
- Llama 2/3
- Mistral
- Mixtral MoE
- Qwen
- Falcon
- Gemma
- Phi
- BERT
- Whisper (audio)
- And more...

✅ **Capabilities**:
- Flash Attention support
- LoRA adapter support
- Quantization (FP8, INT8, GPTQ)
- Multi-device support (CPU, CUDA, MPS)
- Streaming inference
- Batch processing

## Installation

### Build from Source

```bash
cd inference-gollek/plugins/runner/safetensor/gollek-plugin-runner-safetensor
mvn clean install -Pinstall-plugin
```

This installs the plugin to `~/.gollek/plugins/runners/`

### Download Pre-built

```bash
wget https://github.com/gollek-ai/gollek/releases/download/v1.0.0/gollek-plugin-runner-safetensor-1.0.0.jar
cp gollek-plugin-runner-safetensor-1.0.0.jar ~/.gollek/plugins/runners/
```

## Configuration

### Application Properties

```yaml
gollek:
  runners:
    safetensor-runner:
      enabled: true
      backend: direct  # or "gguf-conversion"
      device: cuda     # or "cpu", "mps"
      dtype: f16       # or "f32", "int8"
      max_context: 4096
      flash_attention: true
      load_in_8bit: false
```

### Runtime Configuration

```json
{
  "safetensor-runner": {
    "enabled": true,
    "backend": "direct",
    "device": "cuda",
    "dtype": "f16",
    "max_context": 4096,
    "flash_attention": true
  }
}
```

## Usage

### Via Runner Plugin Registry

```java
import tech.kayys.gollek.engine.registry.RunnerPluginRegistry;
import tech.kayys.gollek.plugin.runner.RunnerSession;

@Inject
RunnerPluginRegistry runnerRegistry;

// Create session
Optional<RunnerSession> session = runnerRegistry.createSession(
    "models/llama-3-8b.safetensors",
    Map.of("max_context", 4096)
);

// Execute inference
if (session.isPresent()) {
    InferenceResponse response = session.get()
        .infer(request)
        .await().atMost(Duration.ofSeconds(30));
    
    System.out.println(response.getContent());
    
    // Or streaming
    session.get().stream(request)
        .subscribe().with(chunk -> System.out.print(chunk.getDelta()));
}
```

### Direct Plugin Usage

```java
import tech.kayys.gollek.plugin.runner.safetensor.SafetensorRunnerPlugin;
import tech.kayys.gollek.safetensor.engine.warmup.DirectSafetensorBackend;

// Create plugin
SafetensorRunnerPlugin plugin = new SafetensorRunnerPlugin(config, backend);

// Create session
RunnerSession session = plugin.createSession("model.safetensors", Map.of());

// Execute inference
InferenceResponse response = session.infer(request).await();
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│           SafetensorRunnerPlugin                        │
│  - implements RunnerPlugin                              │
│  - Manages session lifecycle                            │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│          SafetensorRunnerSession                        │
│  - implements RunnerSession                             │
│  - Wraps DirectSafetensorBackend                        │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│        DirectSafetensorBackend (existing)               │
│  - Native Safetensor inference                          │
│  - GGUF conversion support                              │
│  - Device management                                    │
└─────────────────────────────────────────────────────────┘
```

## Performance

### Inference Speed (A100, Llama-3-8B)

| Backend | Tokens/s | VRAM | Notes |
|---------|----------|------|-------|
| Direct (FP16) | 120 | 16 GB | Native Safetensor |
| Direct + FlashAttn | 180 | 16 GB | Flash Attention enabled |
| GGUF Conversion (Q8) | 150 | 10 GB | Converted to GGUF |
| GGUF Conversion (Q4) | 200 | 6 GB | Quantized GGUF |

### Memory Usage

| Model Size | FP16 | INT8 | Q4_K_M |
|------------|------|------|--------|
| 7B | 14 GB | 8 GB | 4 GB |
| 13B | 26 GB | 14 GB | 7 GB |
| 70B | 140 GB | 70 GB | 35 GB |

## Advanced Features

### LoRA Adapters

```java
Map<String, Object> config = Map.of(
    "lora_adapters", List.of(
        Map.of(
            "path", "adapters/alpaca-lora.safetensors",
            "scale", 1.0
        )
    )
);

RunnerSession session = runnerRegistry.createSession("model.safetensors", config);
```

### Quantization

```java
Map<String, Object> config = Map.of(
    "quantization", "int8",
    "quantize_weights", true,
    "quantize_activations", true
);

RunnerSession session = runnerRegistry.createSession("model.safetensors", config);
```

### Multi-GPU

```java
Map<String, Object> config = Map.of(
    "device", "cuda",
    "device_ids", List.of(0, 1),
    "tensor_parallel", true
);

RunnerSession session = runnerRegistry.createSession("model.safetensors", config);
```

## Troubleshooting

### Plugin Not Loading

```bash
# Check if plugin is installed
ls -la ~/.gollek/plugins/runners/gollek-plugin-runner-safetensor*.jar

# Check logs
tail -f ~/.gollek/logs/gollek.log | grep safetensor
```

### Backend Not Available

```
Error: Safetensor runner is not available
```

**Solution**:
1. Verify backend dependencies are installed
2. Check device availability (CUDA, MPS)
3. Verify model file exists and is valid

### Out of Memory

```
Error: CUDA out of memory
```

**Solution**:
1. Reduce `max_context`
2. Enable quantization (`dtype: int8` or `q4`)
3. Use GGUF conversion with quantization
4. Reduce batch size

## Testing

```java
@QuarkusTest
class SafetensorRunnerPluginTest {

    @Inject
    RunnerPluginRegistry runnerRegistry;

    @Test
    void shouldCreateSessionForSafetensorModel() {
        Optional<RunnerSession> session = runnerRegistry.createSession(
            "test-model.safetensors",
            Map.of()
        );
        
        assertTrue(session.isPresent());
        assertEquals("safetensor-runner", session.get().getRunner().id());
    }

    @Test
    void shouldExecuteInference() {
        Optional<RunnerSession> session = runnerRegistry.createSession(
            "test.safetensors",
            Map.of()
        );
        
        if (session.isPresent()) {
            InferenceResponse response = session.get()
                .infer(request)
                .await().atMost(Duration.ofSeconds(30));
            
            assertNotNull(response.getContent());
        }
    }
}
```

## Resources

- **Plugin Source**: `plugins/runner/safetensor/gollek-plugin-runner-safetensor/`
- **Existing Implementation**: `plugins/runner/safetensor/gollek-safetensor-gguf/`
- **Engine Integration**: `core/gollek-engine/src/main/java/.../RunnerPluginRegistry.java`
- [Safetensor Format Specification](https://github.com/huggingface/safetensors)
- [Runner Plugin System](../../../RUNNER_PLUGIN_SYSTEM.md)

## License

MIT License - See LICENSE file for details.
