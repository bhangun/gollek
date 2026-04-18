# gollek-quantizer-gptq

GPTQ implementation for Gollek — Hessian-based one-shot weight quantization for LLMs at 2/3/4/8 bits.

This module provides a production-grade GPTQ quantization and dequantization engine for Java, utilizing the **JDK 25 Vector API** for high-performance SIMD-accelerated inference.

## Features

- **Hessian-based Quantization**: Implements the full GPTQ algorithm for minimal accuracy loss during 4-bit compression.
- **SIMD Acceleration**: Leverages Java's Vector API (AVX2, AVX-512, NEON) for ultra-fast weight dequantization and matrix-vector operations.
- **Auto-detection**: Automatically detects GPTQ configurations from HuggingFace/AutoGPTQ safetensor metadata.
- **Memory Efficiency**: Uses off-heap memory (Foreign Function & Memory API) for loading multi-gigabyte models without GC overhead.
- **Format Compatibility**: Fully compatible with standard `.safetensors` models from AutoGPTQ and HuggingFace.

## Algorithm

GPTQ quantizes each weight matrix W layer-by-layer using second-order information from the Hessian of the layer's input activations.

### Citation

```bibtex
@article{frantar2022gptq,
  title   = {GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers},
  author  = {Elias Frantar and Saleh Ashkboos and Torsten Hoefler and Dan Alistarh},
  journal = {arXiv preprint arXiv:2210.17323},
  year    = {2022},
  url     = {https://arxiv.org/abs/2210.17323}
}
```

### Dequantization Formula

For each element `w[i,j]` in a group `g`:
```
w̃ = (unpack(qweight) − unpack(qzeros)) · scales
```

## Performance & Vector API

This module requires **Java 25+** to utilize the **JDK Vector API**. The dequantization engine (`VectorDequantizer`) uses SIMD instructions to process multiple output features in parallel:

- **AVX-512**: Processes 16 floats per iteration (512-bit).
- **AVX2**: Processes 8 floats per iteration (256-bit).
- **NEON**: Processes 4 floats per iteration (128-bit).

### Enabling Vector API

When running with this module, ensure you add the following JVM flags:
```bash
--add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED
```

## Usage

### Java API

```java
try (GPTQQuantizerService service = new GPTQQuantizerService()) {
    // Quantize FP32 model → GPTQ INT4
    QuantizationResult result = service.quantize(
        Path.of("model/"),
        Path.of("model-gptq/"),
        GPTQConfig.gptq4bit()
    );

    // Load and dequantize
    GPTQLoader loader = service.loadQuantized(Path.of("model-gptq/"));
    
    // Inspect model
    ModelInspectionResult info = service.inspect(Path.of("model-gptq/"));
    System.out.println("Layers: " + info.layerCount());
}
```

### JBang Example

A ready-to-run example is available in the `examples/jbang` directory:

```bash
# Run with synthetic demo data
jbang gollek/examples/jbang/quantizer/gollek-quantizer-gptq.java --demo

# Quantize a real model
jbang gollek/examples/jbang/quantizer/gollek-quantizer-gptq.java \
  --model /path/to/model/dir \
  --bits 4 --group-size 128
```

## Compression

| bits | Elements per INT32 | Compression vs F32 |
|------|-------------------|-------------------|
| 2    | 16                | 16×               |
| 3    | 10 (+ 2 padding)  | ~10.7×            |
| 4    | 8                 | 8×                |
| 8    | 4                 | 4×                |

## Module Structure

| Class | Role |
|-------|------|
| `GPTQConfig` | Configuration record — bits, groupSize, actOrder, symmetric, exllamaV2 |
| `GPTQQuantizerService` | High-level quantize / dequantize / inspect service |
| `GPTQLoader` | Load GPTQ safetensors, auto-detect config |
| `GPTQSafetensorConverter` | GPTQ → FP32/FP16 dequantization converter |
| `VectorDequantizer` | JDK Vector API dequantization engine |
| `MemoryAllocator` | Off-heap memory management via FFM API |
| `QuantizedLayer` | Per-layer quantized weight container |

## Related Quantizers

- **`gollek-quantizer-awq`**: Activation-aware Weight Quantization.
- **`gollek-quantizer-autoround`**: Advanced sign-SGD based optimization.
- **`gollek-quantizer-quip`**: High-compression 2-bit lattice quantization.
