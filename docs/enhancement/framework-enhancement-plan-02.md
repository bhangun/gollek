# Gollek Framework Enhancement Plan v2.0 — Part 2 (JDK 25 Edition)

**Document Version:** 2.0  
**Date:** April 9, 2026  
**Continues From:** `framework-enhancement-plan-01.md`  
**Status:** 🎯 Planning Phase

### What's New in v2.0

This revision updates the framework plan with **JDK 25 modern capabilities**:

✨ **Foreign Function & Memory (FFM) API** — Replace JNI with safe, fast native interop
- 30-50% faster than JNI (JIT-compiled downcalls)
- Zero-copy memory exchange with CUDA/Metal
- Automatic memory safety via Arena allocation
- No JNI boilerplate or C++ compiler needed

✨ **Vector API (SIMD on CPU)** — 3-5x speedup for tensor operations
- Auto-vectorization for AVX-512, AVX-2, NEON
- 4-6x faster dot products, softmax, layer-norm
- JIT compiler generates optimal code
- Portable across Intel, AMD, ARM architectures

✨ **Records & Sealed Classes** — Type-safe immutable data types
- Zero Lombok overhead
- Automatic equals/hashCode/toString
- Sealed hierarchies for Device, Backend types
- Better IDE support and error messages

✨ **Enhanced Pattern Matching** — More elegant switch expressions
- Device type routing without boilerplate
- Type patterns for tensor operations
- Cleaner control flow

---

## Overview

Part 1 covered the high-level 4-phase roadmap. This document dives deeper into:

1. **SDK Unification** — clean developer surface across all runtimes
2. **NLP Ecosystem** — tokenization, training, fine-tuning
3. **Hardware Acceleration** — GPU/Metal binding strategy
4. **Implementation Details** — concrete class designs and APIs
5. **Integration with Wayang Platform** — agent, RAG, workflow hooks

---

## 5. SDK Unification (Critical Gap)

### Problem

Currently the SDK has fragmented surfaces:
- Different runners (GGUF, ONNX, LiteRT, LibTorch) expose different APIs
- No unified session abstraction
- KV cache / streaming / batching not unified
- Hard to plug into agent framework

### Target: Unified Client API

```java
// Single entry point regardless of backend
GollekClient client = GollekClient.builder()
    .endpoint("http://localhost:8080")
    .model("qwen2-7b")
    .build();

// Streaming-first
client.generateStream(Prompt.text("Explain attention"))
      .onToken(System.out::print)
      .onComplete(result -> log(result.metrics()));

// Batch inference
List<GenerationResult> results = client.generateBatch(prompts);

// Embeddings
float[] embedding = client.embed("Hello world");
```

### New Module: `gollek-sdk-unified`

```
gollek/sdk/lib/gollek-sdk-unified/
├── GollekClient.java              # Main entry point
├── GollekClientBuilder.java
├── session/
│   ├── InferenceSession.java      # Session abstraction
│   ├── TrainingSession.java
│   └── SessionConfig.java
├── api/
│   ├── GenerationRequest.java
│   ├── GenerationResult.java
│   ├── GenerationStream.java      # Streaming response
│   ├── EmbeddingRequest.java
│   └── BatchRequest.java
├── routing/
│   ├── BackendRouter.java         # Routes to GGUF/ONNX/etc.
│   ├── FeatureNegotiator.java     # Detects backend capabilities
│   └── LoadBalancer.java
└── README.md
```

### Core Interfaces

**`GollekClient.java`**
```java
public interface GollekClient {
    GenerationResult generate(GenerationRequest request);
    GenerationStream generateStream(GenerationRequest request);
    List<GenerationResult> generateBatch(List<GenerationRequest> requests);
    float[] embed(String text);
    ModelInfo modelInfo();
    void close();

    static GollekClientBuilder builder() { ... }
}
```

**`InferenceSession.java`**
```java
public interface InferenceSession extends AutoCloseable {
    GenerationResult run(GenerationRequest request);
    GenerationStream stream(GenerationRequest request);
    SessionMetrics metrics();
    boolean supportsFeature(Feature feature);  // KV_CACHE, SPECULATIVE, MULTIMODAL
}
```

**`GenerationStream.java`**
```java
public interface GenerationStream {
    GenerationStream onToken(Consumer<String> handler);
    GenerationStream onComplete(Consumer<GenerationResult> handler);
    GenerationStream onError(Consumer<Throwable> handler);
    CompletableFuture<GenerationResult> toFuture();
    Stream<String> toStream();  // Java Stream API
}
```

### Feature Negotiation

```java
public enum Feature {
    KV_CACHE,
    SPECULATIVE_DECODING,
    MULTIMODAL,
    FLASH_ATTENTION,
    QUANTIZATION_INT8,
    STREAMING,
    BATCH_INFERENCE
}

// Auto-detected at session creation
FeatureSet features = session.supportedFeatures();
if (features.has(Feature.KV_CACHE)) {
    session.enableKVCache(KVCacheConfig.defaults());
}
```

**Deliverables:**
- [ ] `GollekClient` unified interface
- [ ] `InferenceSession` abstraction
- [ ] `GenerationStream` reactive API
- [ ] Backend router (GGUF/ONNX/LiteRT/LibTorch)
- [ ] Feature negotiation
- [ ] Migration guide from old APIs
- [ ] Examples for each backend

**Timeline:** Phase 1, Weeks 1-6 (parallel with CNN/RNN)

---

## 6. NLP Ecosystem (Critical Gap)

### 6.1 Tokenizer Framework

**Problem:** No public tokenization API for training. Inference works but training requires tokenizing datasets natively.

**New Module:** `gollek-sdk-tokenizer` (enhance existing `gollek-tokenizer-core`)

```
gollek/sdk/lib/gollek-sdk-tokenizer/
├── Tokenizer.java                 # Core interface
├── TokenizerConfig.java
├── impl/
│   ├── BPETokenizer.java          # Byte-Pair Encoding
│   ├── WordPieceTokenizer.java    # BERT-style
│   ├── SentencePieceTokenizer.java
│   └── TikTokenTokenizer.java     # GPT-style
├── loader/
│   ├── HuggingFaceTokenizerLoader.java  # Load tokenizer.json
│   └── VocabLoader.java
└── README.md
```

**Core Interface:**
```java
public interface Tokenizer {
    List<Integer> encode(String text);
    String decode(List<Integer> ids);
    BatchEncoding batchEncode(List<String> texts, int maxLength);
    int vocabSize();
    int padTokenId();
    int eosTokenId();
    int bosTokenId();
}
```

**HuggingFace Compatibility:**
```java
// Load any HuggingFace tokenizer
Tokenizer tokenizer = HuggingFaceTokenizerLoader
    .from(Path.of("tokenizer.json"));

// Or from Hub
Tokenizer tokenizer = ModelHub.loadTokenizer("bert-base-uncased");

// Batch encode for training
BatchEncoding batch = tokenizer.batchEncode(
    texts,
    maxLength = 512,
    padding = true,
    truncation = true
);
```

**Deliverables:**
- [ ] `Tokenizer` interface
- [ ] BPE, WordPiece, SentencePiece implementations
- [ ] HuggingFace `tokenizer.json` loader
- [ ] Padding/truncation/attention masks
- [ ] Example: BERT fine-tuning pipeline
- [ ] Unit tests (>95% coverage)

**Timeline:** Phase 1, Weeks 5-10

---

### 6.2 Model StateDict & Serialization

**Problem:** Can load SafeTensors but cannot save trained models.

**Enhance:** `gollek-sdk-nn`

```java
// Save
model.saveStateDict(Path.of("model.safetensors"));
model.save(Path.of("checkpoint/"));  // Full checkpoint

// Load
model.loadStateDict(Path.of("model.safetensors"));
Module loaded = Module.load(Path.of("checkpoint/"));

// Partial loading (transfer learning)
model.loadStateDict(pretrained, strict = false);

// Inspect
Map<String, Tensor> stateDict = model.stateDict();
stateDict.forEach((name, param) -> 
    System.out.printf("%s: %s%n", name, param.shape()));
```

**StateDict API:**
```java
public interface StateDictCapable {
    Map<String, GradTensor> stateDict();
    void loadStateDict(Map<String, GradTensor> state);
    void loadStateDict(Map<String, GradTensor> state, boolean strict);
    void saveStateDict(Path path);
    void save(Path checkpointDir);
    static Module load(Path checkpointDir);
}
```

**Deliverables:**
- [ ] `stateDict()` / `loadStateDict()` on all modules
- [ ] SafeTensors save/load
- [ ] GGUF save/load
- [ ] Checkpoint with optimizer state
- [ ] Strict/non-strict loading
- [ ] Example: Fine-tune and save BERT
- [ ] Unit tests

**Timeline:** Phase 1, Weeks 8-12

---

### 6.3 NLP Training Pipeline

**Enhance:** `gollek-sdk-nlp`

```
gollek/sdk/lib/gollek-sdk-nlp/
├── TextDataset.java               # Text dataset with tokenization
├── TextClassificationTrainer.java
├── SequenceToSequenceTrainer.java
├── LanguageModelTrainer.java      # CLM / MLM
├── tasks/
│   ├── TextClassification.java
│   ├── TokenClassification.java   # NER
│   ├── QuestionAnswering.java
│   └── Summarization.java
└── README.md
```

**Example: Fine-tune for classification:**
```java
// Dataset
TextDataset dataset = TextDataset.builder()
    .tokenizer(tokenizer)
    .texts(texts)
    .labels(labels)
    .maxLength(512)
    .build();

// Model
BertForSequenceClassification model =
    BertForSequenceClassification.pretrained("bert-base-uncased", numLabels=2);

// Trainer
TextClassificationTrainer trainer = TextClassificationTrainer.builder()
    .model(model)
    .optimizer(AdamW.create(model.parameters(), lr=2e-5, weightDecay=0.01))
    .epochs(3)
    .warmupSteps(100)
    .build();

trainer.fit(dataset.split(0.8));
```

**Deliverables:**
- [ ] `TextDataset` with tokenization
- [ ] Fine-tuning trainers for common tasks
- [ ] Warmup LR scheduler
- [ ] Example: Sentiment analysis fine-tuning
- [ ] Example: NER fine-tuning
- [ ] Documentation

**Timeline:** Phase 2, Months 7-9

---

## 7. Hardware Acceleration Strategy (JDK 25 FFM + Vector API)

### 7.1 Device Abstraction

**Problem:** `GradTensor` uses `float[]` — all ops on CPU. Need transparent GPU support.

**Design:**
```java
// Device placement (PyTorch-style)
GradTensor tensor = GradTensor.of(data).to(Device.CUDA);
GradTensor tensor = GradTensor.of(data).to(Device.METAL);  // Apple
GradTensor tensor = GradTensor.of(data).to(Device.CPU);

// Model to device
model.to(Device.CUDA);

// Auto device selection
Device device = Device.best();  // CUDA > Metal > CPU
```

**`Device.java`:**
```java
public sealed interface Device permits Device.CPU, Device.CUDA, Device.Metal, Device.ROCm {
    record CPU() implements Device {}
    record CUDA(int index) implements Device {
        static CUDA of(int index) { return new CUDA(index); }
    }
    record Metal() implements Device {}

    static Device best() {
        if (CUDARuntime.isAvailable()) return new CUDA(0);
        if (MetalRuntime.isAvailable()) return new Metal();
        return new CPU();
    }
}
```

### 7.2 JDK 25 Foreign Function & Memory (FFM) Integration

**Approach:** Replace JNI with JDK 25 FFM API for safer, faster native interop. Use Panama Project FFM.

**Benefits:**
- ✅ No JNI boilerplate, no `native` declarations
- ✅ JIT-compiled FFM bridges (faster than JNI)
- ✅ Automatic memory safety with Arena allocation
- ✅ Zero-copy data exchange with native libraries
- ✅ Easy CUDA/cuDNN/Metal bindings

**`CudaMemory.java` (FFM-based):**
```java
import java.lang.foreign.*;

public class CudaMemory {
    static final Linker LINKER = Linker.nativeLinker();
    static final SymbolLookup CUDA_LOOKUP = LINKER.defaultLookup();

    // Function pointers for cuDNN/cuBLAS operations
    static final FunctionDescriptor MATMUL_DESC = FunctionDescriptor.ofVoid(
        ValueLayout.JAVA_LONG,  // handle
        ValueLayout.JAVA_INT,   // m
        ValueLayout.JAVA_INT,   // n
        ValueLayout.JAVA_INT,   // k
        ValueLayout.ADDRESS,    // A
        ValueLayout.ADDRESS     // B
    );

    static final MemorySegment MATMUL = LINKER.downcallHandle(
        CUDA_LOOKUP.find("cublasSgemm").orElseThrow(),
        MATMUL_DESC
    );

    // Arena-allocated device memory (auto-freed)
    public static float[] deviceMatmul(float[] a, float[] b, int m, int n, int k) {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment aSegment = arena.allocateArray(ValueLayout.JAVA_FLOAT, a);
            MemorySegment bSegment = arena.allocateArray(ValueLayout.JAVA_FLOAT, b);
            MemorySegment cSegment = arena.allocate(
                ValueLayout.JAVA_FLOAT.byteSize() * m * n
            );

            // Call native cuBLAS
            MATMUL.invoke(cudaHandle(), m, n, k, aSegment, bSegment, cSegment);

            return cSegment.toArray(ValueLayout.JAVA_FLOAT);
        }
    }
}
```

**Module-info.java (FFM requires explicit permission):**
```java
module gollek.sdk.tensor {
    requires java.base;
    opens tech.kayys.gollek.ml.tensor to java.base;
    
    // JDK 25: Enable FFM for CUDA/Metal/ROCm
    opens tech.kayys.gollek.ml.tensor.ffm to jdk.incubator.foreign;
}
```

### 7.3 Vector API (JDK 25) for Vectorized Operations

**JDK 25 Vector API enables SIMD acceleration on CPU (AVX-512, NEON):**

**`VectorizedOps.java` (using Vector API):**
```java
import jdk.incubator.vector.*;

public class VectorizedOps {
    static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    // Vectorized element-wise add: ~4x faster than scalar loop
    public static float[] vectorAdd(float[] a, float[] b) {
        float[] result = new float[a.length];
        int limit = SPECIES.loopBound(a.length);

        for (int i = 0; i < limit; i += SPECIES.length()) {
            var va = FloatVector.fromArray(SPECIES, a, i);
            var vb = FloatVector.fromArray(SPECIES, b, i);
            va.add(vb).intoArray(result, i);
        }

        // Scalar tail for remainder
        for (int i = limit; i < a.length; i++) {
            result[i] = a[i] + b[i];
        }
        return result;
    }

    // Vectorized dot product: Critical for attention mechanism
    public static float vectorDot(float[] a, float[] b) {
        FloatVector acc = FloatVector.zero(SPECIES);
        int limit = SPECIES.loopBound(a.length);

        for (int i = 0; i < limit; i += SPECIES.length()) {
            var va = FloatVector.fromArray(SPECIES, a, i);
            var vb = FloatVector.fromArray(SPECIES, b, i);
            acc = acc.add(va.mul(vb));
        }

        float result = acc.reduceLanes(VectorOperators.ADD);

        // Scalar tail
        for (int i = limit; i < a.length; i++) {
            result += a[i] * b[i];
        }
        return result;
    }

    // Vectorized softmax: Essential for transformer layers
    public static float[] vectorSoftmax(float[] x) {
        float[] exp = new float[x.length];
        float[] result = new float[x.length];

        // Find max for numerical stability
        float max = Float.NEGATIVE_INFINITY;
        for (float v : x) max = Math.max(max, v);

        // Vectorized exp(x - max)
        float finalMax = max;
        int limit = SPECIES.loopBound(x.length);

        for (int i = 0; i < limit; i += SPECIES.length()) {
            var v = FloatVector.fromArray(SPECIES, x, i);
            var shifted = v.sub(finalMax);
            // Approximate exp using polynomial or lookup table
            v.exp().intoArray(exp, i);
        }

        for (int i = limit; i < x.length; i++) {
            exp[i] = (float) Math.exp(x[i] - finalMax);
        }

        // Sum for normalization
        FloatVector sumAcc = FloatVector.zero(SPECIES);
        for (int i = 0; i < limit; i += SPECIES.length()) {
            sumAcc = sumAcc.add(FloatVector.fromArray(SPECIES, exp, i));
        }
        float sum = sumAcc.reduceLanes(VectorOperators.ADD);

        // Add scalar tail
        for (int i = limit; i < x.length; i++) {
            sum += exp[i];
        }

        // Vectorized division
        for (int i = 0; i < limit; i += SPECIES.length()) {
            FloatVector.fromArray(SPECIES, exp, i)
                .div(sum)
                .intoArray(result, i);
        }

        for (int i = limit; i < x.length; i++) {
            result[i] = exp[i] / sum;
        }

        return result;
    }
}
```

**Integration into `GradTensor`:**
```java
public class GradTensor {
    // Use Vector API for CPU, FFM for GPU
    private MemorySegment data;
    private Device device;

    public GradTensor add(GradTensor other) {
        if (device == Device.CPU) {
            return new GradTensor(
                VectorizedOps.vectorAdd(this.toArray(), other.toArray()),
                Device.CPU
            );
        } else if (device == Device.CUDA(0)) {
            return new GradTensor(
                CudaMemory.deviceAdd(this.data, other.data),
                device
            );
        }
        throw new IllegalArgumentException("Unsupported device");
    }
}
```

### 7.4 Backend Binding Strategy (FFM-based)

**Approach:** Use JDK 25 FFM for safe, performant bindings to CUDA/Metal.

```
gollek/sdk/lib/gollek-sdk-tensor/
├── storage/
│   ├── TensorStorage.java         # Interface
│   ├── HeapStorage.java           # float[] (current)
│   ├── CUDAStorage.java           # CUDA device memory (FFM-based)
│   └── MetalStorage.java          # Apple Metal (FFM-based)
├── ops/
│   ├── TensorOps.java             # Interface
│   ├── CPUOps.java                # Pure Java + Vector API
│   ├── VectorizedOps.java         # JDK 25 Vector API (AVX-512, NEON)
│   ├── CudaOps.java               # cuBLAS/cuDNN via FFM
│   └── MetalOps.java              # Metal Performance Shaders via FFM
├── ffm/
│   ├── CudaMemory.java            # FFM CUDA bindings
│   ├── MetalMemory.java           # FFM Metal bindings
│   └── NativeLibraryLoader.java   # Smart library loading
└── Device.java
```

**FFM Compilation Requirements:**
```xml
<build>
    <plugins>
        <plugin>
            <groupId>org.apache.maven.plugins</groupId>
            <artifactId>maven-compiler-plugin</artifactId>
            <version>3.13.0</version>
            <configuration>
                <source>25</source>
                <target>25</target>
                <!-- Enable JDK 25 preview features -->
                <compilerArgs>
                    <arg>--enable-preview</arg>
                    <arg>--release</arg>
                    <arg>25</arg>
                </compilerArgs>
            </configuration>
        </plugin>
    </plugins>
</build>

<properties>
    <argLine>--enable-preview --add-modules jdk.incubator.foreign</argLine>
</properties>
```

### 7.4 Phased GPU Rollout (with JDK 25 FFM)

**Phase 2 (Months 7-9): Foundation**
- Device abstraction API
- CPU/GPU memory management
- JDK 25 FFM bridges to CUDA/Metal (replaces JNI)
- Vector API for CPU SIMD (AVX-512, NEON)
- `tensor.to(device)` semantics

**Phase 3 (Months 13-15): Acceleration**
- cuBLAS matmul via FFM (JIT-compiled, faster than JNI)
- cuDNN Conv2d, BatchNorm via FFM
- Flash Attention CUDA kernel integration
- Apple Metal Performance Shaders via FFM
- Vector API optimization for softmax, attention

**Phase 4 (Months 19-21): Advanced**
- TensorRT integration via FFM
- ROCm for AMD via FFM
- Custom CUDA kernels with FFM interop
- Multi-GPU memory management with Arena allocation

**Deliverables (Phase 2):**
- [ ] `Device` sealed interface
- [ ] `TensorStorage` abstraction with FFM MemorySegment
- [ ] JDK 25 FFM bridge to CUDA (cuBLAS stubs)
- [ ] JDK 25 Vector API ops (add, dot, softmax)
- [ ] `tensor.to(Device.CUDA)` working
- [ ] CPU vectorization benchmark: scalar vs Vector API
- [ ] FFM memory safety validation
- [ ] Documentation with JDK 25 setup guide

### 7.5 Performance Expectations (JDK 25 FFM + Vector API)

| Operation | Scalar | Vector API | GPU (FFM) |
|-----------|--------|-----------|-----------|
| MatMul 512x512 | 1.0x | 3.5-4.0x | 50-100x |
| Softmax | 1.0x | 4.5-5.5x | 30-50x |
| Dot Product | 1.0x | 3.0-3.5x | 20-40x |
| Conv2d 3x3 | 1.0x | 2.0-2.5x | 40-80x |
| Attention | 1.0x | 3.0-4.0x | 50-100x |

**Key Advantages of FFM over JNI:**
- ✅ 30-50% faster than JNI (JIT-compiled downcall handles)
- ✅ No allocation overhead with Arena
- ✅ Memory safety (controlled by JVM)
- ✅ Zero-copy interop with native arrays
- ✅ Better error handling (no segfaults)

---

## 8. JDK 25 Modern Features & Capabilities

This section details how to leverage JDK 25's latest features for Gollek SDK.

### 8.1 Foreign Function & Memory (FFM) API

**What:** Safe, performant native code interoperability without JNI boilerplate.

**Benefits for Gollek:**
- Direct bindings to CUDA, Metal, ROCm, cuDNN, cuBLAS
- Arena-based memory allocation (automatic cleanup)
- JIT-compiled downcall handles (faster than JNI)
- Pointer-safe through MemorySegment
- Zero hidden allocations

**Example: CUDA Matrix Multiplication via FFM**

```java
import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;

public class CudaBlas {
    // CUDA handle type
    static final class CudaHandle {}

    // Function signature for cublasSgemm
    static final FunctionDescriptor SGEMM_DESC = FunctionDescriptor.ofVoid(
        ValueLayout.JAVA_INT,      // transa
        ValueLayout.JAVA_INT,      // transb
        ValueLayout.JAVA_INT,      // m
        ValueLayout.JAVA_INT,      // n
        ValueLayout.JAVA_INT,      // k
        ValueLayout.JAVA_FLOAT,    // alpha
        ValueLayout.ADDRESS,       // A
        ValueLayout.JAVA_INT,      // lda
        ValueLayout.ADDRESS,       // B
        ValueLayout.JAVA_INT,      // ldb
        ValueLayout.JAVA_FLOAT,    // beta
        ValueLayout.ADDRESS,       // C
        ValueLayout.JAVA_INT       // ldc
    );

    static final Linker LINKER = Linker.nativeLinker();
    static final SymbolLookup CUBLAS = LINKER.defaultLookup();

    static final MethodHandle SGEMM = LINKER.downcallHandle(
        CUBLAS.find("cublasSgemm").orElseThrow(),
        SGEMM_DESC
    );

    // Safe matrix multiplication
    public static float[] matmul(float[] a, float[] b, int m, int n, int k) {
        try (Arena arena = Arena.ofConfined()) {
            // Allocate and copy to GPU memory via FFM
            MemorySegment aSegment = arena.allocateArray(ValueLayout.JAVA_FLOAT, a);
            MemorySegment bSegment = arena.allocateArray(ValueLayout.JAVA_FLOAT, b);
            MemorySegment cSegment = arena.allocateArray(
                ValueLayout.JAVA_FLOAT, 
                new float[m * n]
            );

            // Call native function (JIT-compiled by JVM)
            SGEMM.invoke(
                0, 0,  // transa, transb
                m, n, k,
                1.0f,  // alpha
                aSegment, k,
                bSegment, n,
                0.0f,  // beta
                cSegment, n
            );

            return cSegment.toArray(ValueLayout.JAVA_FLOAT);
        } catch (Throwable e) {
            throw new RuntimeException("CUDA matmul failed", e);
        }
    }
}
```

**Comparison: JNI vs FFM**

| Feature | JNI | FFM |
|---------|-----|-----|
| Boilerplate | High (native + Java glue) | Minimal |
| Performance | Slower (crossing 2 boundaries) | Faster (JIT-compiled downcalls) |
| Memory Safety | Unsafe (manual allocation) | Safe (Arena + MemorySegment) |
| Setup | Complex (C++ compiler) | Simple (Java only) |
| Inlining | Limited | Excellent (inlined by JVM) |

### 8.2 Vector API (SIMD on CPU)

**What:** Portable, efficient SIMD operations for CPUs (AVX-512, AVX-2, NEON, etc.).

**Benefits for Gollek:**
- 3-5x speedup on element-wise ops
- 4-6x speedup on reductions (sum, dot product)
- Auto-vectorizes softmax, attention, layer-norm
- No JNI overhead
- JIT compiler can generate optimal code

**Core Operations Candidates:**

| Operation | Speedup | Critical For |
|-----------|---------|--------------|
| Element-wise add/multiply | 3.5-4.0x | All layers |
| Dot product | 4.0-5.0x | Attention |
| Softmax | 4.5-5.5x | Transformers |
| Layer normalization | 4.0-4.5x | All models |
| Matrix reduction | 3.5-4.0x | Loss computation |

**Practical Implementation:**

```java
import jdk.incubator.vector.*;

public class VectorizedMath {
    static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    // Fast softmax with Vector API
    public static void softmax(float[] x) {
        // 1. Find max for stability
        float max = Float.NEGATIVE_INFINITY;
        for (float v : x) max = Math.max(max, v);

        // 2. Compute exp(x - max) with vector operations
        float[] exp = new float[x.length];
        computeExp(x, max, exp);

        // 3. Vectorized sum for normalization
        float sum = vectorSum(exp);

        // 4. Vectorized division
        vectorDivide(exp, sum, x);
    }

    private static void computeExp(float[] x, float max, float[] exp) {
        int limit = SPECIES.loopBound(x.length);

        for (int i = 0; i < limit; i += SPECIES.length()) {
            var v = FloatVector.fromArray(SPECIES, x, i);
            var shifted = v.sub(max);
            shifted.exp().intoArray(exp, i);
        }

        // Scalar tail for non-aligned remainder
        for (int i = limit; i < x.length; i++) {
            exp[i] = (float) Math.exp(x[i] - max);
        }
    }

    private static float vectorSum(float[] arr) {
        FloatVector acc = FloatVector.zero(SPECIES);
        int limit = SPECIES.loopBound(arr.length);

        for (int i = 0; i < limit; i += SPECIES.length()) {
            var v = FloatVector.fromArray(SPECIES, arr, i);
            acc = acc.add(v);
        }

        float result = acc.reduceLanes(VectorOperators.ADD);

        // Scalar tail
        for (int i = limit; i < arr.length; i++) {
            result += arr[i];
        }

        return result;
    }

    private static void vectorDivide(float[] src, float divisor, float[] dst) {
        int limit = SPECIES.loopBound(src.length);

        for (int i = 0; i < limit; i += SPECIES.length()) {
            var v = FloatVector.fromArray(SPECIES, src, i);
            v.div(divisor).intoArray(dst, i);
        }

        for (int i = limit; i < src.length; i++) {
            dst[i] = src[i] / divisor;
        }
    }
}
```

### 8.3 Records & Sealed Classes

**Already using in Gollek SDK:**

```java
// Sealed class hierarchy for Device types
public sealed interface Device permits Device.CPU, Device.CUDA, Device.Metal {
    record CPU() implements Device {}
    record CUDA(int index) implements Device {}
    record Metal() implements Device {}
}

// Immutable data transfer objects
public record GenerationResult(
    String text,
    int tokenCount,
    long latencyMs,
    float[] logits
) {}

public record AttentionOutput(
    GradTensor output,
    GradTensor[] weights,
    GradTensor cache
) {}
```

**Benefits:**
- Concise immutable types
- Automatic equals/hashCode/toString
- Type-safe sealed hierarchies
- No Lombok overhead

### 8.4 Pattern Matching (Enhanced)

**JDK 25 improves pattern matching for more elegant code:**

```java
// Tensor operation routing
public GradTensor operate(Device device, String op, GradTensor x) {
    return switch (device) {
        case Device.CPU cpu -> {
            // Inline pattern with binding
            yield switch (op) {
                case "softmax" -> VectorizedOps.softmax(x);
                case "add" -> CPUOps.add(x, x);
                default -> throw new UnsupportedOperationException(op);
            };
        }
        case Device.CUDA cuda -> CudaOps.compute(cuda.index(), op, x);
        case Device.Metal metal -> MetalOps.compute(op, x);
    };
}

// Type pattern matching for tensor operations
public float computeStatistic(GradTensor tensor) {
    return tensor match {
        case Vector2D v -> v.norm();
        case MatrixND m when m.shape().length == 2 -> m.frobeniusNorm();
        case MatrixND m -> m.spectralNorm();
        case _ -> 0.0f;
    };
}
```

### 8.5 JDK 25 Compilation & Runtime

**Update pom.xml:**

```xml
<project>
    <properties>
        <java.version>25</java.version>
        <maven.compiler.source>25</maven.compiler.source>
        <maven.compiler.target>25</maven.compiler.target>
    </properties>

    <build>
        <plugins>
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-compiler-plugin</artifactId>
                <version>3.13.0</version>
                <configuration>
                    <source>25</source>
                    <target>25</target>
                    <!-- Enable JDK 25 preview features -->
                    <compilerArgs>
                        <arg>--enable-preview</arg>
                        <arg>--add-modules=jdk.incubator.foreign</arg>
                    </compilerArgs>
                </configuration>
            </plugin>

            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-surefire-plugin</artifactId>
                <version>3.2.2</version>
                <configuration>
                    <argLine>--enable-preview --add-modules jdk.incubator.foreign</argLine>
                </configuration>
            </plugin>
        </plugins>
    </build>

    <dependencies>
        <!-- JDK 25 FFM stubs (if needed) -->
        <dependency>
            <groupId>org.openjdk.jdk.panama</groupId>
            <artifactId>panama-foreign</artifactId>
            <version>25</version>
            <scope>provided</scope>
        </dependency>
    </dependencies>
</project>
```

**Module-info.java (required for FFM):**

```java
module gollek.sdk.core {
    requires java.base;
    
    // FFM bindings module
    requires jdk.incubator.foreign;
    
    // Enable FFM for this module
    opens tech.kayys.gollek.ml.tensor.ffm 
        to jdk.incubator.foreign;
    
    exports tech.kayys.gollek.ml;
    exports tech.kayys.gollek.ml.autograd;
    exports tech.kayys.gollek.ml.nn;
    exports tech.kayys.gollek.ml.tensor;
}
```

**Runtime JVM Args (for testing/development):**

```bash
# Enable Vector API
java -XX:+UnlockExperimentalVMOptions \
     -XX:+UseVectorCmove \
     -XX:+UseAVX=3 \
     --enable-preview \
     --add-modules jdk.incubator.foreign \
     -jar gollek-sdk-benchmark.jar

# For Apple Silicon (use NEON)
java --enable-preview \
     -XX:+UseNeon \
     --add-modules jdk.incubator.foreign \
     -cp gollek-sdk-*.jar \
     TestVectorizedOps
```

---

## 9. Tensor Operations — Concrete Implementation

### 8.1 Missing Operations in `GradTensor`

**Slicing & Indexing:**
```java
// Slice along dimension
GradTensor seq = tensor.slice(dim=0, start=0, end=10);

// Boolean masking
GradTensor masked = tensor.maskedSelect(mask);
tensor.maskedFill_(mask, value=0.0f);

// Gather/scatter (critical for attention, loss functions)
GradTensor gathered = tensor.gather(dim=1, index);
tensor.scatter_(dim=1, index, src);

// Advanced indexing
GradTensor selected = tensor.index(new int[]{0, 2, 4});
```

**Shape Operations:**
```java
// Concatenate and stack
GradTensor cat = GradTensor.cat(List.of(a, b, c), dim=0);
GradTensor stacked = GradTensor.stack(List.of(a, b, c), dim=0);

// Split
List<GradTensor> chunks = tensor.split(chunkSize=64, dim=0);
List<GradTensor> parts = tensor.chunk(numChunks=4, dim=0);

// Squeeze/unsqueeze
GradTensor expanded = tensor.unsqueeze(dim=0);
GradTensor squeezed = tensor.squeeze(dim=0);
```

**Reduction Operations:**
```java
tensor.sum(dim=1, keepdim=true)
tensor.mean(dim=1)
tensor.max(dim=1)
tensor.min(dim=1)
tensor.argmax(dim=1)
tensor.std(dim=1)
tensor.var(dim=1)
tensor.norm(p=2, dim=1)
```

**Einstein Summation:**
```java
// Batch matmul
GradTensor result = GradTensor.einsum("bij,bjk->bik", a, b);

// Attention scores
GradTensor scores = GradTensor.einsum("bhid,bhjd->bhij", q, k);

// Outer product
GradTensor outer = GradTensor.einsum("i,j->ij", a, b);
```

**Deliverables:**
- [ ] `slice`, `gather`, `scatter`, `maskedSelect`
- [ ] `cat`, `stack`, `split`, `chunk`
- [ ] Full reduction ops with `dim` and `keepdim`
- [ ] `einsum` with common patterns
- [ ] Backward pass for all new ops
- [ ] Unit tests (>95% coverage)

**Timeline:** Phase 1, Weeks 17-22

---

## 10. Integration with Wayang Platform

### 9.1 Agent Integration

The Gollek SDK should integrate cleanly with `wayang-gollek/agent`:

```java
// Agent using Gollek SDK for inference
@Agent
public class InferenceAgent {

    @Inject
    GollekClient client;

    @Skill("generate")
    public String generate(String prompt) {
        return client.generate(
            GenerationRequest.of(prompt)
        ).text();
    }

    @Skill("embed")
    public float[] embed(String text) {
        return client.embed(text);
    }
}
```

**New Module:** `gollek-sdk-agent-integration`

```
gollek/sdk/lib/gollek-sdk-agent-integration/
├── AgentGollekClient.java         # Agent-aware client
├── SkillInferenceAdapter.java     # Skill → inference bridge
├── MemoryAwareSession.java        # Session with agent memory
└── README.md
```

### 9.2 RAG Integration

```java
// RAG pipeline using SDK
RAGPipeline pipeline = RAGPipeline.builder()
    .embedder(client.asEmbedder())
    .vectorStore(vectorStore)
    .generator(client.asGenerator())
    .build();

String answer = pipeline.query("What is attention mechanism?");
```

### 9.3 Gamelan Workflow Integration

```java
// Workflow node using Gollek SDK
@WorkflowNode("inference")
public class InferenceNode implements Node {

    @Override
    public NodeResult execute(NodeContext ctx) {
        String prompt = ctx.input("prompt");
        GenerationResult result = client.generate(
            GenerationRequest.of(prompt)
                .maxTokens(512)
                .temperature(0.7f)
        );
        return NodeResult.of("output", result.text());
    }
}
```

**Deliverables:**
- [ ] Agent integration module
- [ ] RAG pipeline adapter
- [ ] Gamelan workflow node
- [ ] Examples for each integration
- [ ] Documentation

**Timeline:** Phase 2, Months 10-12

---

## 11. Developer Experience (DX) Improvements

### 10.1 Fluent Builder APIs

All SDK components should use consistent fluent builders:

```java
// Consistent pattern across all modules
Model model = Sequential.builder()
    .add(Linear.of(784, 256))
    .add(ReLU.instance())
    .add(Dropout.rate(0.2f))
    .add(Linear.of(256, 10))
    .build();

Trainer trainer = Trainer.builder()
    .model(model)
    .optimizer(Adam.lr(0.001))
    .loss(CrossEntropyLoss.instance())
    .epochs(100)
    .callbacks(EarlyStopping.patience(5), ModelCheckpoint.best())
    .build();
```

### 10.2 JBang Examples

Provide runnable JBang scripts for quick experimentation:

```java
///usr/bin/env jbang "$0" "$@" ; exit $?
//DEPS tech.kayys.gollek:gollek-sdk-nn:0.2.0
//DEPS tech.kayys.gollek:gollek-sdk-train:0.2.0

import tech.kayys.gollek.ml.nn.*;
import tech.kayys.gollek.sdk.train.*;

public class QuickStart {
    public static void main(String[] args) {
        var model = new Sequential(new Linear(2, 4), new ReLU(), new Linear(4, 1));
        var trainer = Trainer.builder().model(model).epochs(100).build();
        trainer.fit(xorDataset());
    }
}
```

### 10.3 Annotation-Based Model Definition

```java
// Declarative model definition
@Model
public class MyClassifier extends Module {

    @Layer
    private final Linear fc1 = Linear.of(784, 256);

    @Layer
    private final ReLU relu = ReLU.instance();

    @Layer
    private final Linear fc2 = Linear.of(256, 10);

    @Override
    public GradTensor forward(GradTensor x) {
        return fc2.forward(relu.forward(fc1.forward(x)));
    }
}
```

### 10.4 Python Interop

Bridge for teams migrating from Python:

```java
// Load PyTorch model weights
Module model = PytorchBridge.loadModel("model.pt");

// Convert numpy arrays
GradTensor tensor = NumpyBridge.fromNpy("data.npy");

// Export to Python-compatible format
model.saveStateDict("weights.safetensors");  // Loadable in Python
```

**Deliverables:**
- [ ] Consistent builder APIs across all modules
- [ ] 20+ JBang runnable examples
- [ ] Annotation-based model definition
- [ ] PyTorch weight loading
- [ ] NumPy array bridge
- [ ] Interactive REPL support

**Timeline:** Phase 1-2 (continuous)

---

## 12. Testing Strategy

### 11.1 Test Pyramid

```
gollek/sdk/lib/*/src/test/
├── unit/           # Fast, isolated, >95% coverage
├── integration/    # Cross-module, medium speed
├── benchmark/      # JMH benchmarks
└── e2e/            # Full pipeline tests
```

### 11.2 Numerical Correctness Tests

All autograd operations must be verified against PyTorch:

```java
@Test
void testLinearBackward() {
    // Compare gradient with PyTorch reference values
    GradTensor x = GradTensor.of(new float[]{1, 2, 3, 4}, new int[]{2, 2});
    GradTensor w = GradTensor.of(new float[]{0.1f, 0.2f, 0.3f, 0.4f}, new int[]{2, 2});

    GradTensor out = x.matmul(w);
    out.sum().backward();

    // Reference gradients computed in PyTorch
    assertArrayEquals(new float[]{0.3f, 0.7f, 0.3f, 0.7f}, x.grad(), 1e-5f);
}
```

### 11.3 Performance Benchmarks (JMH)

```java
@Benchmark
@BenchmarkMode(Mode.Throughput)
public void matmulBenchmark(Blackhole bh) {
    GradTensor a = GradTensor.randn(512, 512);
    GradTensor b = GradTensor.randn(512, 512);
    bh.consume(a.matmul(b));
}
```

### 11.4 Regression Tests

- Track performance across releases
- Alert on >5% regression
- Compare with PyTorch baseline

**Deliverables:**
- [ ] Unit test suite per module (>90% coverage)
- [ ] Numerical correctness tests vs PyTorch
- [ ] JMH benchmark suite
- [ ] CI/CD integration
- [ ] Performance dashboard

---

## 13. Release & Versioning Strategy

### 12.1 Semantic Versioning

```
0.x.y  — Current (breaking changes allowed)
1.0.0  — Phase 1 complete (stable API)
2.0.0  — Phase 2 complete (production ready)
3.0.0  — Phase 3 complete (advanced features)
```

### 12.2 Module Versioning

All SDK modules versioned together via BOM:

```xml
<!-- Single BOM import -->
<dependencyManagement>
    <dependencies>
        <dependency>
            <groupId>tech.kayys.gollek</groupId>
            <artifactId>gollek-bom</artifactId>
            <version>1.0.0</version>
            <type>pom</type>
            <scope>import</scope>
        </dependency>
    </dependencies>
</dependencyManagement>

<!-- Then use without versions -->
<dependency>
    <groupId>tech.kayys.gollek</groupId>
    <artifactId>gollek-sdk-nn</artifactId>
</dependency>
```

### 12.3 Deprecation Policy

- Deprecated APIs kept for 2 major versions
- Migration guides for every breaking change
- `@Deprecated` + `@since` + `@see` Javadoc

---

## 14. Updated Module Map

```
gollek/sdk/lib/
├── gollek-sdk-tensor/         ← Enhance: advanced ops, device abstraction
├── gollek-sdk-autograd/       ← Enhance: higher-order grads, hooks
├── gollek-sdk-nn/             ← Enhance: statedict, more layers
├── gollek-sdk-cnn/            ← NEW: Conv2d, MaxPool, etc.
├── gollek-sdk-rnn/            ← NEW: LSTM, GRU
├── gollek-sdk-transformers/   ← NEW: Transformer blocks
├── gollek-sdk-train/          ← Enhance: distributed, gradient accum
├── gollek-sdk-data/           ← Enhance: augmentation, streaming
├── gollek-sdk-tokenizer/      ← NEW: BPE, WordPiece, HF loader
├── gollek-sdk-nlp/            ← Enhance: fine-tuning tasks
├── gollek-sdk-vision/         ← Enhance: image transforms
├── gollek-sdk-multimodal/     ← Enhance: vision-language
├── gollek-sdk-optimize/       ← Enhance: quantization, pruning
├── gollek-sdk-distributed/    ← NEW: DDP, multi-node
├── gollek-sdk-viz/            ← NEW: TensorBoard, W&B
├── gollek-sdk-models/         ← NEW: pre-trained zoo
├── gollek-sdk-unified/        ← NEW: unified client API
├── gollek-sdk-agent-integration/ ← NEW: agent/RAG/workflow hooks
├── gollek-sdk-ml/             ← Enhance: sklearn-style API
├── gollek-sdk-hub/            ← Enhance: model registry
├── gollek-sdk-export/         ← Enhance: ONNX, mobile
├── gollek-sdk-litert/         ← Existing: LiteRT/TFLite
└── gollek-sdk-onnx/           ← Existing: ONNX runtime
```

---

## 15. Immediate Next Steps

### Week 1-2 Actions

1. **Tensor ops gap analysis** — list every missing op vs PyTorch
2. **Conv2d design doc** — im2col vs direct convolution decision
3. **Tokenizer loader** — prototype HuggingFace `tokenizer.json` parser
4. **StateDict API** — design and implement `Module.save/load`
5. **Device abstraction** — design `Device` sealed interface

### Milestone: v0.2.0 (End of Phase 1)

- [ ] Conv2d + MaxPool2d working
- [ ] LSTM working
- [ ] Tokenizer loading HuggingFace models
- [ ] `model.save()` / `model.load()` working
- [ ] Advanced tensor ops (slice, gather, cat, stack)
- [ ] Unified client API
- [ ] 15+ JBang examples
- [ ] All tests passing

---

**Document Owner:** Gollek Core Team  
**Last Updated:** April 7, 2026  
**Next:** `framework-enhancement-plan-03.md` — Implementation walkthroughs
