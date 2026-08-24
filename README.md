<div align="center">
  <img src="https://raw.githubusercontent.com/kayys-tech/repo-assets/master/gollek03%404x.png" alt="Gollek" width="300" />
  
  # Gollek Inference Engine
  
  [![Java 25](https://img.shields.io/badge/Java-25-blue.svg)](https://jdk.java.net/25/)
  [![Gradle](https://img.shields.io/badge/Gradle-9.x-brightgreen.svg)](https://gradle.org/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Native Image](https://img.shields.io/badge/GraalVM-Native%20Image-orange.svg)](https://www.graalvm.org/)
</div>

**Gollek** is a high-performance, low-level inference engine and serving layer for AI models. It is the unified evolution of the former `alkhawarizm` (tensor/model runtime) and `tafkir` (quantization) projects, now consolidated as a single standalone project.

Gollek acts as the foundation for the [Wayang](../wayang/) orchestration layer. While Wayang handles agents and orchestration, Gollek handles the raw inference machinery: model loading, tensor operations, quantization, GGUF/SafeTensor execution, and hardware-accelerated backends (CPU, Metal, CUDA).

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Wayang (Agent Layer)              │
│          Orchestration · RAG · MCP · Tools          │
└───────────────────────┬─────────────────────────────┘
                        │  gollek-sdk / gollek-sdk-api
┌───────────────────────▼─────────────────────────────┐
│              Gollek (Inference Engine)              │
│                                                     │
│  ┌──────────┐  ┌──────────┐  ┌───────────────────┐ │
│  │  GGUF    │  │SafeTensor│  │  LiteRT / ONNX    │ │
│  │  Runner  │  │  Runner  │  │     Runners       │ │
│  └──────────┘  └──────────┘  └───────────────────┘ │
│                                                     │
│  ┌──────────────────────────────────────────────┐  │
│  │         alkhawarizm (tensor core)            │  │
│  │  CPU · Metal (Apple Silicon) · CUDA          │  │
│  └──────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────┘
```

## Prerequisites

- **Java 25** (GraalVM for native image, Temurin for JVM)
- **Gradle 9.x** (wrapper included)
- **macOS** (Apple Silicon, Metal acceleration) or **Linux** (CPU / CUDA)

Optional:
- `graalvm-ce-25` for `buildNative`
- `llama.cpp` source (auto-prepared by `scripts/prepare-llama-source.sh`)

## Quick Start

### Install via script

```bash
curl -sSL https://raw.githubusercontent.com/kayys-tech/gollek/main/scripts/install.sh | bash
```

### Build from source

```bash
git clone https://github.com/kayys-tech/gollek.git
cd gollek

# Build JVM jar
./gradlew :ui:gollek-cli:quarkusBuild -x test

# Build native binary (requires GraalVM 25)
./gradlew :ui:gollek-cli:buildNative -x test
```

### Run the CLI

```bash
# Interactive chat
gollek chat --model <modelId>

# Pull a model from HuggingFace (task type auto-detected)
gollek pull hf:Qwen/Qwen2.5-7B-Instruct-GGUF

# Pull a vision model
gollek pull hf:HuggingFaceTB/SmolVLM-256M-Instruct

# List models with task-type grouping
gollek list

# Filter by task category
gollek list --task-type vision
gollek list --task-type stt

# Single-shot inference (clean output for scripts)
gollek run --model <id> --no-banner --no-info --only-text --prompt "Hello"

# Start the API server
gollek serve --model <id> --port 8080

# Check system info and resource utilization
gollek system info
```

## CLI Reference

See the full [CLI README](ui/gollek-cli/README.md) for all commands and flags. Key highlights:

### Model task-type grouping

Every model downloaded via `gollek pull` is automatically tagged with a **task type** derived from the
HuggingFace `pipeline_tag`:

| HuggingFace tag | Gollek task type |
|-----------------|------------------|
| `text-generation`, `conversational` | `text` |
| `automatic-speech-recognition`, `audio-classification` | `stt` |
| `text-to-speech`, `text-to-audio` | `tts` |
| `image-to-text`, `image-classification`, `object-detection` | `vision` |
| `feature-extraction`, `sentence-similarity` | `embedding` |

```bash
gollek list              # shows TASK column
gollek list -t vision    # filter by category
gollek pull hf:myorg/my-tts-model --task-type tts  # override
```

### Clean output flags for scripting

```bash
gollek run --model <id> \
  --no-banner   \   # suppress ASCII banner
  --no-info     \   # suppress model/provider line
  --only-text   \   # print only the assistant text
  --prompt "..."
```

## SDK Usage

```java
// Embedded SDK with built-in resource metrics
GollekSdk sdk = GollekSdkFactory.create(
    GollekSdkConfig.local()
        .observability(SdkObservabilityProvider.withResourceMetrics())
        .build()
);

// Infer
InferenceResponse response = sdk.infer("qwen2.5-0.5b", request);

// Check resource utilization (for capacity planning)
SdkMetricsCollector.ResourceSnapshot resources = sdk.getResourceSnapshot();
System.out.printf("CPU: %.1f%% | Heap: %dMB / %dMB%n",
    resources.processCpuLoad() * 100,
    resources.heapUsedBytes() / 1_048_576,
    resources.heapMaxBytes() / 1_048_576);
```

## Observability

Gollek exposes metrics for capacity planning via:

| Metric | Description |
|--------|-------------|
| `gollek.resource.cpu.process.load` | JVM process CPU load (0-1) |
| `gollek.resource.cpu.system.load` | System-wide CPU load (0-1) |
| `gollek.resource.memory.heap.used` | JVM heap used (bytes) |
| `gollek.resource.memory.heap.max` | JVM heap max (bytes) |
| `gollek.inference.request.cpu.seconds` | CPU time per request (histogram) |
| `gollek.inference.request.memory.delta` | Heap delta per request (histogram) |
| `gollek.inference.duration` | Request latency (histogram) |
| `gollek.inference.ttft` | Time-to-first-token (histogram) |
| `gollek.inference.tpot` | Time-per-output-token (histogram) |

Metrics are exported via Micrometer (Prometheus-compatible) when running as a server.

## Logging

Gollek writes structured JSON logs to `~/.gollek/logs/gollek.log` by default.

```bash
# Override log directory
export GOLLEK_LOG_DIR=/var/log/gollek
gollek start
```

The JSON format is compatible with Fluentd, Logstash, Loki, and Elasticsearch.

## Project Structure

```
Families/gollek/
├── framework/
│   ├── core/           # gollek-core: inference orchestration
│   ├── observability/  # Metrics, tracing, logging (Micrometer + OTel)
│   ├── sdk/            # gollek-sdk, gollek-sdk-core, gollek-sdk-api
│   └── spi/            # Service Provider Interfaces
├── runtime/
│   ├── plugins/        # Dynamic runner plugins
│   └── runner/         # GGUF, SafeTensor, LiteRT, ONNX runners
├── ui/
│   └── gollek-cli/     # Quarkus-based CLI (JVM + native image)
├── scripts/            # install.sh, prepare-llama-source.sh, etc.
└── docs/               # Extended documentation
```

## Docs

- [Quickstart](docs/QUICKSTART.md)
- [API Quick Reference](docs/API_QUICK_REFERENCE.md)
- [Observability Guide](docs/OBSERVABILITY.md)
- [API Examples](docs/API_EXAMPLES.md)
- [Error Codes](docs/error-codes.md)
- [Install Profiles](docs/install-profiles.md)

## License

MIT © Kayys.tech
