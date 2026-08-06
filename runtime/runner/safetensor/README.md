# Gollek SafeTensor - Modular Inference Engine

**Production-ready Safetensors inference engine with modular architecture**

---

## 📦 Module Overview

The SafeTensor inference engine is organized into focused, concern-specific modules for better maintainability and clearer dependencies.

```
inference-gollek/extension/runner/safetensor/
├── 📋 REORGANIZATION_PLAN.md    # Detailed reorganization guide
├── 📋 README.md                  # This file
├── 📋 pom.xml                    # Parent POM
├── ├── gollek-safetensor-api/          # Public API & SPI
├── ├── gollek-safetensor-loader/       # Safetensor file format
├── ├── gollek-safetensor-core/         # Core inference engine
├── ├── gollek-safetensor-text/         # Text models
├── ├── gollek-safetensor-audio/        # Audio (TTS/STT)
├── ├── gollek-safetensor-vision/       # Vision & multimodal
├── ├── gollek-safetensor-rag/          # RAG & vector DB
├── ├── gollek-safetensor-tooling/      # Tool calling
├── ├── gollek-safetensor-session/      # Session management
├── ├── gollek-safetensor-routing/      # Model routing
└── └── gollek-safetensor-integration/  # Integration tests
```

---

## 🎯 Module Descriptions

### Core Infrastructure

| Module | Description | Dependencies |
|--------|-------------|--------------|
| **api** | Public API, SPI interfaces, common types | None |
| **loader** | Safetensor file format parsing & weight loading | api |
| **core** | Core inference engine, generation, lifecycle | api, loader |

### Domain-Specific Modules

| Module | Description | Dependencies |
|--------|-------------|--------------|
| **text** | Text model architectures, tokenization | api, core |
| **audio** | TTS (SpeechT5), STT (Whisper) | api, core |
| **vision** | Vision encoding, multimodal inference | api, core |

### Feature Modules

| Module | Description | Dependencies |
|--------|-------------|--------------|
| **rag** | RAG pipeline, Qdrant integration | api, core, text |
| **tooling** | Tool calling, function execution | api, core |
| **session** | Conversation session management | api, core |
| **routing** | Model routing, multi-tenancy, A/B testing | api, core |

### Testing

| Module | Description | Dependencies |
|--------|-------------|--------------|
| **integration** | Integration tests, end-to-end workflows | All (test scope) |

---

## 🚀 Quick Start

### Build All Modules

```bash
cd inference-gollek/extension/runner/safetensor
mvn clean install
```

### Build Specific Module

```bash
# Build core module only
mvn clean install -pl gollek-safetensor-core -am

# Build audio module
mvn clean install -pl gollek-safetensor-audio -am
```

### Run Tests

```bash
# All modules
mvn test

# Specific module
mvn test -pl gollek-safetensor-core
```

---

## 📋 Usage Examples

### Text Inference (Core + Text)

```java
@Inject DirectInferenceEngine engine;
@Inject TextModelFamilies.LLaMA3Family arch;

// Load model
String key = engine.loadModel(modelPath);

// Generate text
InferenceResponse response = engine.generate(
    "Hello, how are you?",
    modelPath,
    GenerationConfig.builder()
        .maxTokens(100)
        .temperature(0.7)
        .build()
);
```

### Audio TTS (Audio Module)

```java
@Inject SpeechT5Engine tts;

byte[] wav = tts.synthesize(
    "Hello world",
    "alloy",
    modelPath
).await().indefinitely();
```

### RAG (RAG Module)

```java
@Inject QdrantVectorStore qdrant;
@Inject RagPipeline rag;

// Ingest documents
rag.ingest("my-collection", documents, embeddingModel);

// Query
String answer = rag.query(
    "my-collection",
    "What is AI?",
    embeddingModel,
    llmModel
);
```

### Tool Calling (Tooling Module)

```java
@Inject ToolCallingEngine toolCalling;

List<ToolDefinition> tools = List.of(
    ToolDefinition.builder()
        .name("get_weather")
        .description("Get weather for location")
        .build()
);

DetectionResult result = toolCalling.detect(
    modelOutput,
    tools
);
```

---

## 🔧 Module Dependency Graph

```
                    ┌─────────────┐
                    │     API     │
                    └──────┬──────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
    ┌────▼────┐      ┌────▼────┐      ┌────▼────┐
    │ Loader  │      │  Core   │      │  Text   │
    └────┬────┘      └────┬────┘      └────┬────┘
         │                │                │
         │         ┌──────┴──────┐        │
         │         │             │        │
    ┌────▼────┐ ┌──▼──┐   ┌────▼────┐   │
    │  Audio  │ │Vision│   │   RAG   │◄──┘
    └─────────┘ └─────┘   └─────────┘
    
    ┌──────────┐  ┌─────────┐  ┌─────────┐
    │ Tooling  │  │ Session │  │ Routing │
    └──────────┘  └─────────┘  └─────────┘
```

---

## 📊 Module Statistics

| Module | Java Files | Key Components |
|--------|------------|----------------|
| **api** | 5-8 | ModelArchitecture, ModelConfig, GenerationConfig |
| **loader** | 15-20 | SafetensorFFMLoader, SafetensorDType |
| **core** | 20-25 | DirectForwardPass, FlashAttentionKernel, MoE |
| **text** | 20-25 | TextModelFamilies (10+ architectures) |
| **audio** | 3-5 | SpeechT5Engine, WhisperEngine |
| **vision** | 5-7 | VisionEncoder, MultimodalInferenceEngine |
| **rag** | 5-7 | QdrantVectorStore, RagPipeline |
| **tooling** | 3-5 | ToolCallingEngine |
| **session** | 2-3 | ConversationSessionManager |
| **routing** | 5-7 | ModelRouter, MultiTenantQuotaService |
| **integration** | 5-10 | Integration tests |

**Total:** ~90-120 Java files

---

## 🎯 Key Features

### Core Inference
- ✅ DirectForwardPass (prefill + decode)
- ✅ FlashAttention with GQA & RoPE
- ✅ KV Cache management
- ✅ Beam search decoding
- ✅ Logprobs computation
- ✅ MoE support (Mixtral, DeepSeek-MoE)

### Model Architectures (20+ families)
- ✅ LLaMA / LLaMA-2 / LLaMA-3
- ✅ Gemma / Gemma-2 / Gemma-3
- ✅ Qwen-2 / Qwen-2.5 / Qwen-3
- ✅ Mistral / Mixtral / Mistral-Small
- ✅ Yi, Cohere, DeepSeek, Phi, SmolLM

### Audio
- ✅ SpeechT5 TTS (6 voices)
- ✅ Whisper STT
- ✅ Audio REST endpoints

### Vision
- ✅ CLIP ViT encoder
- ✅ Multimodal inference (LLaVA, Qwen-VL, etc.)
- ✅ Vision model registry

### RAG
- ✅ Qdrant vector database integration
- ✅ In-memory retrieval fallback
- ✅ RAG pipeline with chunking

### Tool Calling
- ✅ 5 format detection (LLaMA-3.1, Qwen2, Mistral, etc.)
- ✅ System prompt generation
- ✅ Parallel tool calls

### Session Management
- ✅ Multi-turn conversation
- ✅ KV cache persistence
- ✅ Sliding window context

### Routing & Multi-Tenancy
- ✅ A/B testing
- ✅ Canary deployments
- ✅ Shadow mode
- ✅ Per-tenant quotas

---

## 🔨 Development

### Adding a New Module

1. Create module directory structure:
```bash
mkdir -p gollek-safetensor-{name}/src/main/java/tech/kayys/gollek/inference/safetensor/{package}
mkdir -p gollek-safetensor-{name}/src/test/java
```

2. Create POM with dependencies:
```xml
<parent>
    <groupId>tech.kayys.gollek</groupId>
    <artifactId>gollek-safetensor-parent</artifactId>
    <version>1.0.0-SNAPSHOT</version>
</parent>

<artifactId>gollek-safetensor-{name}</artifactId>
```

3. Add module to parent POM

### Module Best Practices

1. **Single Responsibility:** Each module should have one clear purpose
2. **Minimal Dependencies:** Only depend on what you need
3. **Clear API:** Expose only necessary public interfaces
4. **Test Coverage:** Include unit and integration tests
5. **Documentation:** Document public APIs and usage

---

## 📝 Reorganization

For details on the reorganization from the monolithic structure, see:
- `REORGANIZATION_PLAN.md` - Detailed reorganization guide
- `reorganize.sh` - Automated migration script

---

## 🚀 Build & Deployment

### Local Development

```bash
# Clean build
mvn clean install

# Skip tests for faster builds
mvn clean install -DskipTests

# Build specific module + dependencies
mvn clean install -pl gollek-safetensor-audio -am
```

### Docker Deployment

```bash
# Build Docker image
docker build -f Dockerfile.safetensor -t gollek-safetensor:latest .

# Run with Docker Compose
docker-compose -f docker-compose.safetensor.yml up -d
```

---

## 📚 Documentation

- `README.md` - This overview
- `REORGANIZATION_PLAN.md` - Reorganization details
- `CHANGES.md` - Version history (in each module)
- `application.properties.example` - Configuration reference

---

## 🎉 Summary

The Gollek SafeTensor modular architecture provides:

✅ **Clear Separation of Concerns** - Each module has a single responsibility  
✅ **Better Dependency Management** - Modules only depend on what they need  
✅ **Easier Testing** - Smaller, focused modules  
✅ **Improved Maintainability** - Changes are isolated  
✅ **Flexible Deployment** - Include only needed modules  
✅ **Better Documentation** - Focused per-module docs  

**Total:** 11 focused modules, ~100 Java files, production-ready!

---

**Reorganization Date:** 2026-03-20  
**Status:** ✅ Complete  
**Next:** Build and verify all modules
