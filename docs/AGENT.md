# Gollek Inference Engine — Agent Knowledge Item

## What Is Gollek

Gollek is a **pure-Java, native-image-ready LLM inference engine** built on Quarkus 3.32 / Java 25. It runs on Apple Silicon via Apple Accelerate (cblas_sgemm, vDSP), CUDA, ROCm, and DirectML kernels. It ships as both a JVM uber-jar and a GraalVM native CLI (`gollek`).

**Repo root**: `/Users/bhangun/Workspace/workkayys/Products/Wayang/wayang-platform/gollek`  
**Group ID**: `tech.kayys.gollek`  
**Version**: `0.1.0-SNAPSHOT`  
**Installed binary**: `~/.local/bin/gollek` (version `1.0.0`)

---

## Reactor Module Map

```
gollek/
├── pom.xml                         ← Parent POM (Java 25, Quarkus 3.32.2)
├── bom/                            ← Bill of Materials (gollek-bom)
├── core/
│   ├── spi/                        ← SPI interfaces (gollek-spi, gollek-spi-inference, ...)
│   ├── model/                      ← Model architectures (Llama, Qwen, Gemma, Phi, ...)
│   ├── optimization/               ← KV cache, FlashAttention, PagedAttention, prefill/decode
│   └── plugin/                     ← Plugin framework, Metal plugin
├── runner/
│   ├── safetensor/                 ← PRIMARY: pure-Java safetensor inference engine
│   │   ├── gollek-safetensor-core      ← AccelTensor + AccelOps (FFM + Accelerate)
│   │   ├── gollek-safetensor-engine    ← DirectForwardPass, FlashAttentionKernel, TokenSampler
│   │   ├── gollek-safetensor-loader    ← SafetensorFFMLoader, SafetensorShardLoader, LoadCache
│   │   └── gollek-safetensor-tokenizer ← BPE tokenizer
│   ├── llamacpp/                   ← LlamaCpp JNI provider
│   ├── gguf/                       ← Pure-Java Safetensor→GGUF converter
│   ├── onnx/                       ← ONNX runner
│   └── torch/                      ← LibTorch (FFM) runner
├── kernel/
│   ├── metal/                      ← Apple Metal kernel plugin
│   ├── cuda/                       ← CUDA kernel plugin
│   └── rocm/                       ← ROCm kernel plugin
├── inference/
│   ├── gollek-engine               ← Top-level inference orchestrator
│   └── gollek-api-rest             ← REST API (OpenAI-compatible)
├── provider/                       ← Cloud providers (Gemini, OpenAI, Anthropic, Mistral, Cerebras)
├── sdk/                            ← Java SDK (local + remote)
├── plugins/                        ← Feature plugins (RAG, MCP, streaming, quota, ...)
├── ui/
│   └── gollek-cli/                 ← CLI entry point (Picocli + Quarkus)
└── framework/                      ← ML framework (gollek-ml-*)
```

---

## Core Inference Pipeline (Safetensor Engine)

### Forward Pass Flow

```
TokenSampler.sample(logits)
    ↑
DirectForwardPass.decode(tokenId, startPos, weights, config, arch, kvCache)
    → embeddingLookup(embedTable, [tokenId])    → AccelTensor [1, 1, hidden]
    → for each layer i:
        transformerLayer(hidden, weights, config, arch, kvCache, i, startPos, 1)
            → rmsNorm(hidden, attn_norm_weight)
            → FlashAttentionKernel.compute(attnInput)
                → linear(hidden, Q/K/V weights)
                → Q/K norm (Qwen2.5: rmsNorm per head)
                → reshape [B,S,H,D] → transpose(1,2) → contiguous [B,H,S,D]
                → RoPE: applyRopeToSegment (split-half rotation)
                → updateKVCache (PagedAttention block layout)
                → tiledAttention → PagedAttentionVectorAPI.compute
                → transpose(1,2) → reshape → linear(oWeight)
            → residual add
            → rmsNorm(ffn_norm_weight)
            → swigluFfn or ffnNonGated or MoeForwardPass
            → residual add
    → rmsNorm(final_norm_weight)
    → linear(lm_head_weight)        → logits float[]
    → TokenSampler.sample(logits, config, freq)
```

### Key Classes

| Class | Package | Role |
|-------|---------|------|
| `AccelTensor` | `gollek-safetensor-core` | Off-heap float32 tensor (FFM MemorySegment + Arena.ofAuto) |
| `AccelOps` | `gollek-safetensor-core` | SIMD ops: matmul (cblas_sgemm), rmsNorm, softmax, swiglu, silu |
| `DirectForwardPass` | `gollek-safetensor-engine` | Full transformer forward pass (prefill + decode) |
| `FlashAttentionKernel` | `gollek-safetensor-engine` | Multi-head attention with PagedAttention KV cache |
| `RopeFrequencyCache` | `gollek-safetensor-engine` | RoPE frequency precomputation + in-place rotation |
| `PagedAttentionVectorAPI` | `gollek-safetensor-engine` | Paged attention computation (block table reads) |
| `TokenSampler` | `gollek-safetensor-engine` | Greedy / temperature / top-k / top-p / min-p sampling |
| `KVCacheManager` | `gollek-safetensor-engine` | KV cache session management |
| `BlockManager` | `gollek-safetensor-engine` | PagedAttention physical block allocator |
| `SafetensorFFMLoader` | `gollek-safetensor-loader` | mmap-based safetensor file loader via FFM |
| `SafetensorShardLoader` | `gollek-safetensor-loader` | Sharded model loading (model.safetensors.index.json) |
| `SafetensorLoadCache` | `gollek-safetensor-loader` | LRU cache for open load results |
| `SafetensorLoaderFacade` | `gollek-safetensor-loader` | Primary CDI injection point for consumers |
| `BpeTokenizer` | `gollek-safetensor-tokenizer` | BPE tokenization with GGUF space markers (U+0120) |

---

## AccelTensor: Off-Heap Memory Model

```java
// All tensors backed by FFM MemorySegment
// Owning tensor: has Arena.ofAuto() — GC-managed lifetime
// View tensor: shares parent's MemorySegment, no arena

AccelTensor.zeros(long... shape)          // allocate + zero-fill
AccelTensor.fromFloatArray(float[], shape) // copy from heap
AccelTensor.wrapSegment(MemorySegment, shape) // zero-copy wrap (caller owns lifetime)
AccelTensor.copyOf(MemorySegment, shape)  // copy with new arena

t.reshape(newShape)     // zero-copy view (requires contiguous)
t.transpose(dim0, dim1) // zero-copy stride swap
t.contiguous()          // materialize if non-contiguous (allocates new arena)
t.slice(dim, start, end)// zero-copy stride-based slice
t.indexSelect(indices)  // embedding lookup (copies rows)
```

> [!WARNING]
> `AccelTensor.close()` marks the tensor as closed but **does NOT free the underlying arena** (commented out in the source). Arena memory is GC-managed. Do not rely on `close()` for immediate memory reclaim in hot paths.

---

## Model Architecture SPI

```java
// ModelArchitecture defines weight key names per architecture
arch.embedTokensWeight()           // "model.embed_tokens.weight"
arch.lmHeadWeight()                // "lm_head.weight"
arch.finalNormWeight()             // "model.norm.weight"
arch.layerAttentionNormWeight(i)   // "model.layers.{i}.input_layernorm.weight"
arch.layerQueryWeight(i)           // "model.layers.{i}.self_attn.q_proj.weight"
arch.layerKeyWeight(i)             // "model.layers.{i}.self_attn.k_proj.weight"
arch.layerValueWeight(i)           // "model.layers.{i}.self_attn.v_proj.weight"
arch.layerOutputWeight(i)          // "model.layers.{i}.self_attn.o_proj.weight"
arch.layerFfnGateWeight(i)         // "model.layers.{i}.mlp.gate_proj.weight"
arch.layerFfnUpWeight(i)           // "model.layers.{i}.mlp.up_proj.weight"
arch.layerFfnDownWeight(i)         // "model.layers.{i}.mlp.down_proj.weight"
// Qwen2.5-specific:
arch.layerQueryNormWeight(i)       // "model.layers.{i}.self_attn.q_norm.weight"
arch.layerKeyNormWeight(i)         // "model.layers.{i}.self_attn.k_norm.weight"
```

Supported architectures (via `ModelConfig`): **Llama**, **Qwen2/2.5**, **Gemma/Gemma2**, **Phi**, **Mistral**, **DeepSeek** (MoE), **Cohere**.

---

## KV Cache Layout (PagedAttention)

```
Block layout in BlockManager:
  [tokensPerBlock, numKVHeads, headDim * 2]
  where *2 = interleaved K and V

  For token at absPos:
    blockIdxInTable = absPos / tokensPerBlock
    tokenIdxInBlock = absPos % tokensPerBlock
    physicalBlock   = blockTable[blockIdxInTable]

  Within block at (tokenIdx, head, headDim):
    K offset = tokenIdx * numKVHeads * headDim * 2 + head * headDim * 2
    V offset = K offset + headDim
```

---

## Known Inference Bugs

### Token ID 200012 Repetition Bug

**Symptom**: Model generates token ID 200012 repeatedly, producing garbage output.  
**Likely cause**: Logit corruption (NaN/Inf) in attention or residual layers not caught before sampling.  
**Diagnostic**: `DirectForwardPass` has debug output in `prefill()` that prints top-5 token IDs and logit stats (`min/max/sum`) to stderr.

**Mitigation steps**:
1. Check `TokenSampler.sample()` — it validates >10% NaN/Inf and returns -1.
2. Add `checkStability()` calls after each `transformerLayer()` in `DirectForwardPass` — method exists but body is empty.
3. Add a vocab-size clamp in `TokenSampler`: `if (id >= vocabSize) return 0;`
4. Check `RopeFrequencyCache.RopeFrequencies.rotateInPlace` element offset math.

**Status**: Open — needs instrumentation of logit tensor immediately before `TokenSampler.sample()`.

---

## RoPE Implementation

Gollek uses **split-half rotation** (standard for Llama/Qwen2):
```
For position pos, head at elementOffset in [B,H,S,D] layout:
  half = rotaryDim / 2
  For i in [0, half):
    x0 = data[elementOffset + i]
    x1 = data[elementOffset + i + half]
    cos_val = cos[pos][i]
    sin_val = sin[pos][i]
    data[elementOffset + i]        = x0*cos - x1*sin
    data[elementOffset + i + half] = x1*cos + x0*sin
```

`RopeFrequencyCache` is `@ApplicationScoped` and precomputes frequencies up to `maxPositionEmbeddings`.

---

## CDI / Quarkus Patterns

- All core services are `@ApplicationScoped` CDI beans.
- `SafetensorLoaderFacade` is the **primary injection point** for model weight loading — inject this, not `SafetensorFFMLoader` directly.
- `AccelOps` is a **utility class** (no CDI) — use static methods directly.
- `AccelTensor` is a **value class** (no CDI) — instantiate via factory methods.

### Correct injection pattern
```java
@Inject
SafetensorLoaderFacade safetensorLoader;

// Open a session (caller must close)
try (SafetensorShardLoader.SafetensorShardSession session = safetensorLoader.open(modelPath)) {
    AccelTensor weight = session.tensor("model.embed_tokens.weight");
    // ...
}

// Or async
Uni<float[]> weights = safetensorLoader.loadTensorAsync(modelPath, "lm_head.weight",
    tensor -> tensor.toFloatArray());
```

---

## CLI Commands

```bash
gollek run --model qwen2.5-0.5b "Hello"
gollek chat --model qwen2.5-0.5b
gollek pull qwen2.5-0.5b
gollek prepare --model qwen2.5-0.5b         # download + GGUF convert
gollek list
gollek info
gollek extensions
gollek safetensors inspect /path/to/model
gollek providers
gollek --platform metal run --model qwen2.5-0.5b "Hi"
gollek --use-cpu run --model qwen2.5-0.5b "Hi"
```

---

## Java-Specific Constraints

| Concern | Detail |
|---------|--------|
| `Arena.ofShared()` | **BANNED** — breaks GraalVM 25 native-image. Use `Arena.ofAuto()`. |
| `Arena.global()` | OK — used for Accelerate `SymbolLookup` (safe for native-image) |
| `MemorySegment.copy(src, Layout, byteOffset, ...)` | Byte offset must be layout-aligned (multiply element index by `Float.BYTES`) |
| `jdk.incubator.vector` | Incubator module — must be in both compiler args AND native-image args |
| `--enable-preview` | Required for Java 25 features — both JVM and native-image args |
| `quickSortIndices` in TokenSampler | Recursive quicksort on full vocab (150K tokens) — consider partial sort optimization |

---

## Model Path Convention

```
~/.gollek/models/
  gguf/           ← GGUF format models
  safetensor/     ← Safetensors format models
  litert/         ← LiteRT (TF Lite) models
  libtorchscript/ ← LibTorch script models
```

Config key: `gguf.provider.model.base-path=${GOLLEK_HOME:${user.home}/.gollek}/models/gguf`

---

## References

- Skill KI: [`gollek_native_build/artifacts/skill.md`](file:///Users/bhangun/.gemini/antigravity/knowledge/gollek_native_build/artifacts/skill.md)
- Build log analysis: [`native_build_stabilization.md`](file:///Users/bhangun/.gemini/antigravity/brain/9007f635-e3c5-4360-b924-d54ff83d4e4c/artifacts/native_build_stabilization.md)
- Conversation: `9007f635-e3c5-4360-b924-d54ff83d4e4c` (native build + inference debugging)
- Conversation: `ed70b8a6-8b78-4185-a1b5-293a408d470f` (API build stabilization)
- Conversation: `0445ef53-7e41-402e-86fb-a89a32e7b542` (BPE tokenizer + LlamaCpp provider)
