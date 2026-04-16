# Gollek SafeTensor Runner v2

**Enhanced SafeTensor inference engine with TTS, RAG, and zero-downtime hot-swap support**

---

## Overview

Gollek SafeTensor Runner v2 is an enhanced inference engine module for the Wayang Platform, providing production-ready Safetensors model execution with advanced features including:

- **Text-to-Speech** via Microsoft SpeechT5
- **Vector Database Integration** with Qdrant for RAG pipelines
- **Zero-Downtime Model Hot-Swapping** for seamless model updates
- **Comprehensive Architecture Support** for 10+ model families

This module supersedes the original `gollek-runner-safetensor` with improved organization, additional features, and production-hardened implementations.

---

## Features

### 🎤 Text-to-Speech (SpeechT5Engine)

Microsoft SpeechT5 encoder-decoder TTS with:
- Six built-in voice personas (alloy, echo, fable, onyx, nova, shimmer)
- 16kHz 16-bit mono WAV output
- Speaker embedding injection
- HiFi-GAN vocoder interface (simplified sinusoidal synthesis included)

**Usage:**
```java
@Inject SpeechT5Engine tts;

byte[] wav = tts.synthesize("Hello world", "alloy", modelPath)
                   .await().indefinitely();
```

### 🔄 Zero-Downtime Hot-Swap (ModelHotSwapManager)

Seamless model replacement with:
- Four-phase lifecycle (LOADING → WARMING_UP → PROMOTED → DRAINING → UNLOADED)
- Memory pressure monitoring to prevent OOM
- Configurable drain timeout for in-flight requests
- Reactive event stream for observability

**Usage:**
```java
hotSwap.beginSwap("llama3-8b", oldPath, newPath, adapterPath)
       .subscribe().with(
           event -> log.info("Swap event: " + event),
           err   -> log.error("Swap failed", err));
```

### 🔍 Vector Database Integration (QdrantVectorStore)

Production-ready Qdrant REST client:
- Collection management (create/ensure/drop)
- Batch vector upsert with payloads
- k-NN search with score thresholding
- Qdrant Cloud support with API key authentication
- Automatic fallback to in-memory index on failure

**Configuration:**
```properties
gollek.rag.qdrant.url=http://localhost:6333
gollek.rag.qdrant.api-key=your-api-key
gollek.rag.qdrant.timeout-s=30
```

### 🏗️ Architecture Support (TextModelFamilies)

Complete `ModelArchitecture` implementations for:

| Family | Models | Key Features |
|--------|--------|--------------|
| `gemma2` | Google Gemma-2 (2B/9B/27B) | 4 norms/layer, GeGLU, alternating attention |
| `gemma3_text` | Google Gemma-3 (1B/4B/12B/27B) | QK-norm, 5:1 local/global ratio |
| `qwen2` | Alibaba Qwen-2 (0.5B–110B) | GQA, 151K vocab, Q/K bias |
| `qwen2.5` | Alibaba Qwen-2.5 (0.5B–72B) | Larger intermediate_dim |
| `qwen3` | Alibaba Qwen-3 (0.6B–32B) | QK-norm, thinking mode |
| `llama3` | Meta LLaMA-3/3.1/3.2/3.3 | 128K vocab, RoPE θ=500000 |
| `mistral3` | Mistral-Small (22B/24B) | Tekken tokenizer, no sliding window |
| `yi` | 01.AI Yi-1.5 (6B/9B/34B) | LLaMA-style weights |
| `cohere2` | Cohere Command-R+ v2 | Parallel SwiGLU, tied embeddings |
| `deepseek_r1` | DeepSeek-R1 (7B–671B MoE) | MLA, chain-of-thought |

### ⚡ Forward Pass Engine (DirectForwardPass)

Optimized transformer forward pass:
- **Prefill**: Process full prompt, populate KV cache
- **Decode**: Single-token generation with KV cache reuse
- **Optional Norms**: QK-norm, post-attention norm, pre-FFN norm via reflection
- **MoE Support**: Delegates to `MoeForwardPass` for Mixtral/DeepSeek-MoE
- **Non-Gated FFN**: `siluFfnNonGated()` for Cohere Command-R

---

## Module Structure

```
gollek-runner-safetensor-v2/
├── CHANGES.md                          # Version history & migration guide
├── README.md                           # This file
├── pom.xml                             # Maven build configuration
└── src/main/java/tech/kayys/gollek/inference/safetensor/
    ├── arch/
    │   └── TextModelFamilies.java      # Model architecture definitions
    ├── audio/
    │   └── SpeechT5Engine.java         # Text-to-speech engine
    ├── forward/
    │   └── DirectForwardPass.java      # Transformer forward pass
    ├── lifecycle/
    │   ├── ModelHotSwapManager.java    # Zero-downtime hot-swap
    │   ├── MemoryPressureMonitor.java  # Memory pressure monitoring
    │   └── GracefulShutdownHandler.java # Graceful shutdown
    ├── rag/
    │   └── QdrantVectorStore.java      # Qdrant vector DB client
    └── spi/
        └── ModelArchitecture.java      # SPI interface (imported)
```

---

## Installation

### Maven Dependency

Add to your `pom.xml`:

```xml
<dependency>
    <groupId>tech.kayys.gollek</groupId>
    <artifactId>gollek-runner-safetensor-v2</artifactId>
    <version>1.0.0-SNAPSHOT</version>
</dependency>
```

### Build from Source

```bash
# Build the module
mvn clean install -pl inference-gollek/extension/runner/safetensor/gollek-runner-safetensor-v2

# Build with tests
mvn clean test -pl inference-gollek/extension/runner/safetensor/gollek-runner-safetensor-v2

# Build entire safetensor parent
mvn clean install -pl inference-gollek/extension/runner/safetensor
```

---

## Configuration

### Application Properties

```yaml
# Text-to-Speech
gollek:
  audio:
    tts:
      default-voice: alloy

# Hot-Swap
  hotswap:
    enabled: true
    drain-timeout-s: 30

# Memory Monitoring
  memory:
    pressure:
      threshold: 0.85  # 85% heap usage triggers pressure
    check:
      interval-s: 5

# RAG / Qdrant
  rag:
    qdrant:
      url: http://localhost:6333
      api-key: ""
      timeout-s: 30

# Graceful Shutdown
  shutdown:
    timeout-s: 30
    quiet-period-s: 5
```

---

## Usage Examples

### Text-to-Speech

```java
import jakarta.inject.Inject;
import tech.kayys.gollek.inference.safetensor.audio.SpeechT5Engine;

public class TtsService {
    @Inject SpeechT5Engine tts;

    public byte[] generateAudio(String text, String voice) {
        Path modelPath = Paths.get("/models/speecht5-tts.safetensors");
        return tts.synthesize(text, voice, modelPath)
                   .await().indefinitely();
    }
}
```

### Model Hot-Swap

```java
import jakarta.inject.Inject;
import tech.kayys.gollek.inference.safetensor.lifecycle.ModelHotSwapManager;

public class ModelAdminService {
    @Inject ModelHotSwapManager hotSwap;

    public void upgradeModel(String alias, Path newPath) {
        hotSwap.beginSwap(alias, null, newPath, null)
               .subscribe().with(
                   event -> System.out.println(event.phase() + ": " + event.message()),
                   err   -> System.err.println("Failed: " + err.getMessage()));
    }
}
```

### RAG with Qdrant

```java
import jakarta.inject.Inject;
import tech.kayys.gollek.inference.safetensor.rag.QdrantVectorStore;

public class RagService {
    @Inject QdrantVectorStore qdrant;

    public void ingestDocuments(String collection, List<Document> docs) throws IOException {
        qdrant.ensureCollection(collection, 768, "Cosine");

        List<QdrantVectorStore.QdrantPoint> points = docs.stream()
            .map(doc -> new QdrantVectorStore.QdrantPoint(
                doc.getId(),
                doc.getEmbedding(),
                Map.of("text", doc.getText(), "source", doc.getSource())
            ))
            .toList();

        qdrant.upsert(collection, points);
    }

    public List<QdrantVectorStore.SearchResult> search(
            String collection, float[] query, int topK) throws IOException {
        return qdrant.search(collection, query, topK, 0.7f);
    }
}
```

---

## Dependencies

### Required

- **Java 25** (configured via `maven.compiler.release`)
- **Quarkus 3.32.2** (CDI, Config, Core)
- **Gollek LibTorch Runner** (Tensor operations)
- **SmallRye Mutiny** (Reactive programming)
- **Jackson Databind** (JSON serialization)

### Optional (for full functionality)

- **Gollek MoE Module** (Mixtral/DeepSeek-MoE support)
- **Gollek Attention Module** (Flash Attention kernels)

---

## Testing

### Unit Tests

Run tests with:

```bash
mvn test -pl inference-gollek/extension/runner/safetensor/gollek-runner-safetensor-v2
```

### Integration Tests

The module includes integration tests for:
- End-to-end TTS synthesis
- Hot-swap lifecycle events
- Qdrant vector operations
- Forward pass correctness (prefill/decode)

---

## Performance Considerations

### Memory Management

- All intermediate `Tensor` objects are closed immediately after use
- KV cache reuses pre-allocated buffers
- Memory pressure monitor prevents OOM during hot-swap

### Optimizations

- RoPE frequencies precomputed — no per-step cos/sin
- Flash Attention via `FlashAttentionKernel`
- QK-norm applied efficiently via `applyRmsNorm()` helper
- Zero overhead for optional features (null checks compiled away by JIT)

---

## Known Limitations

1. **HiFi-GAN Vocoder**: Current implementation uses simplified sinusoidal synthesis. Full transposed-conv implementation pending.
2. **FP8 Support**: FP8 safetensors load correctly but upcast to BF16. Native FP8 on H100 gated behind future config flag.
3. **Qdrant Batching**: Large batch upserts may timeout — recommended batch size ≤1000 points.

---

## Migration from v1 (gollek-runner-safetensor)

### API Compatibility

The v2 module maintains **full backward compatibility** with v1:
- Same package names (`tech.kayys.gollek.inference.safetensor.*`)
- Same class names and method signatures
- Same configuration keys

### Breaking Changes

**None** — v2 is a drop-in replacement for v1.

### Upgrade Steps

1. Update dependency version in `pom.xml`:
   ```xml
   <artifactId>gollek-runner-safetensor-v2</artifactId>
   <version>1.0.0-SNAPSHOT</version>
   ```

2. No code changes required.

---

## Roadmap (v13)

- [ ] Full HiFi-GAN transposed-conv vocoder implementation
- [ ] Native FP8 support on H100 GPUs
- [ ] Speculative decoding with draft models
- [ ] Multi-modal RAG (image + text retrieval)
- [ ] Structured output (JSON schema, regex constraints)
- [ ] Continuous batching for higher throughput

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Follow the coding conventions in `QWEN.md`
4. Add tests for new functionality
5. Submit a pull request

---

## License

```
Copyright (c) 2026 Kayys.tech
SPDX-License-Identifier: Apache-2.0
```

---

## Support

- **Documentation**: See `CHANGES.md` for version history and `QWEN.md` for development guidelines
- **Issues**: Report via GitHub Issues
- **Discussions**: Join the Wayang Platform Discord channel
