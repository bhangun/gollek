# GGUF Converter — Pure Java, JDK 25 FFM API

Convert HuggingFace model weights (`.safetensors`) to GGUF format using
**zero native code** and the JDK 25 Foreign Function & Memory (FFM) API.

---

## Features

| Feature | Detail |
|---|---|
| **Pure Java** | No JNI, no native libs, no Python dependency |
| **FFM memory-mapped I/O** | `MemorySegment` + `Arena` for zero-copy large tensor reads/writes |
| **GGUF v3** | Full spec compliance: header, KV metadata, tensor descriptors, alignment |
| **Quantization** | F32, F16, BF16→F32, Q8\_0, Q4\_0 (extensible) |
| **Architectures** | LLaMA / Mistral / Qwen2 / Phi / Gemma / Falcon (extensible) |
| **Inspect** | Read and dump any `.gguf` file without external tools |

---

## Requirements

- JDK 25+ (FFM API stable since JDK 22; preview flag used for pattern matching)
- Maven 3.9+

---

## Build

```bash
mvn package -q
# → target/gguf-converter-1.0.0-jar-with-dependencies.jar
```

---

## Usage

### Convert a HuggingFace model directory

```bash
java --enable-preview -jar target/gguf-converter-*-with-dependencies.jar \
  convert /path/to/hf-model-dir output.gguf \
  --type F16 \
  --version 3.1 \
  --verbose
```

**`--type` options:**

| Value | Description |
|---|---|
| `F32` | No quantization (largest, highest quality) |
| `F16` | Half-precision float (default, good balance) |
| `Q8_0` | 8-bit quantization, blocks of 32 |
| `Q4_0` | 4-bit quantization, blocks of 32 |

### Inspect an existing GGUF file

```bash
java --enable-preview -jar target/gguf-converter-*-with-dependencies.jar \
  inspect model.gguf
```

### Run self-tests

```bash
java --enable-preview -cp target/gguf-converter-*-with-dependencies.jar \
  io.gguf.GgufSelfTest
```

---

## Architecture

```
src/main/java/io/gguf/
├── model/
│   ├── GgufModel.java          # In-memory GGUF model (metadata + tensor list)
│   ├── GgufMetaValue.java      # Sealed hierarchy for all KV value types
│   ├── GgufMetaType.java       # GGUF metadata type enum (UINT8 … ARRAY)
│   ├── GgmlType.java           # Tensor element type enum (F32, F16, Q4_0 …)
│   └── TensorInfo.java         # Single tensor descriptor (name, shape, type, offset)
│
├── reader/
│   ├── GgufReader.java         # FFM mmap reader for .gguf files
│   └── SafetensorsReader.java  # FFM mmap reader for .safetensors files
│
├── writer/
│   └── GgufWriter.java         # FFM off-heap writer for .gguf files
│
├── quantize/
│   └── TensorConverter.java    # F16/BF16/Q8_0/Q4_0 conversions via FFM
│
├── arch/
│   ├── HfConfigParser.java     # Parses config.json + tokenizer.json
│   └── LlamaArchMapper.java    # HF tensor names → GGUF names + metadata keys
│
├── HfToGgufConverter.java      # Orchestrator (two-pass: plan → convert → write)
├── GgufSelfTest.java           # Integration tests (no JUnit needed)
└── cli/
    └── GgufConverterMain.java  # CLI entry point
```

### How FFM is used

| Location | FFM Usage |
|---|---|
| `GgufReader` | `FileChannel.map(…, arena)` → `MemorySegment` for zero-copy read |
| `SafetensorsReader` | Same – entire shard mmap'd; `MemorySegment.copy` for tensor slices |
| `GgufWriter.GrowableSegment` | Off-heap `arena.allocate(…)` for metadata buffer; doubles on overflow |
| `TensorConverter` | Off-heap segments for quantization inner loops (avoids GC pressure) |

### GGUF v3 Binary Layout (from spec)

```
 Offset  Size    Field
 ──────  ──────  ─────────────────────────────────────
 0       4       Magic  "GGUF" (0x46554747 LE)
 4       4       Version = 3  (uint32 LE)
 8       8       tensor_count  (uint64 LE)
 16      8       metadata_kv_count  (uint64 LE)
 24      var     metadata KV pairs × metadata_kv_count
 var     var     tensor info descriptors × tensor_count
 var     pad     zero-padding to alignment boundary (default 32 bytes)
 var     data    tensor data blobs (each padded to alignment)
```

Each **KV pair**:
```
  [uint64 key_len][key_len UTF-8 bytes][uint32 type][value bytes]
```

Each **tensor descriptor**:
```
  [uint64 name_len][name bytes][uint32 n_dims][n_dims × uint64 ne][uint32 type][uint64 offset]
```

---

## Extending with New Architectures

1. Add tensor name mappings to `LlamaArchMapper.mapTensorName()`.
2. Add a new `case` in `LlamaArchMapper.mapArch()`.
3. Optionally create a new `XxxArchMapper` for complex remappings.
4. Wire it in `HfToGgufConverter.convert()`.

## Extending with New Quantization Types

1. Add the entry to `GgmlType` enum with correct `blockSize` and `typeSize`.
2. Implement the quantizer method in `TensorConverter`.
3. Add a `case` in the `switch` inside `HfToGgufConverter.convert()`.

---

## References

- [GGUF Spec](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
- [llama.cpp convert_hf_to_gguf.py](https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py)
- [JDK 25 FFM API (JEP 454)](https://openjdk.org/jeps/454)
- [Safetensors format](https://huggingface.co/docs/safetensors)
