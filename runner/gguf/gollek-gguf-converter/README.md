# GGUF Model Converter Service

Enterprise-grade model conversion service for converting AI models to GGUF format using llama.cpp bindings with JDK 25 Foreign Function & Memory (FFM) API.

## Overview

This module provides a production-ready REST API for converting models from various formats (PyTorch, SafeTensors, TensorFlow, Flax) to the efficient GGUF format. It features:

- **Zero-copy native integration** using JDK 25 FFM (no JNI overhead)
- **Multi-tenancy support** with isolated storage and quotas
- **Reactive API** using Mutiny for async operations
- **Real-time progress tracking** via Server-Sent Events
- **Comprehensive quantization options** (F32/F16 to Q2_K)
- **Enterprise features**: metrics, health checks, cancellation
- **Thread-safe** concurrent conversions with resource limits
- **Automatic native library management** with multiple loading strategies

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    REST API Layer                          │
│  (GGUFConverterResource - JAX-RS endpoints)                │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│              Service Layer                                  │
│  (GGUFConverter - Business logic & orchestration)          │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│              FFM Bindings Layer                             │
│  (GGUFNative - Type-safe native function calls)            │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│          C++ Bridge (gguf_bridge.cpp)                       │
│  (Error handling, callbacks, resource management)          │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│              llama.cpp Library                              │
│  (Model conversion & quantization engine)                  │
└─────────────────────────────────────────────────────────────┘
```

## Prerequisites

### Runtime Requirements

- **JDK 25 or later** (with FFM support and preview features enabled)
- **CMake 3.20+** (for building native library)
- **C++17 compiler** (GCC 9+, Clang 10+, or MSVC 2019+)
- **llama.cpp** (cloned as submodule or installed)

### Optional Dependencies

- **CUDA Toolkit 12.0+** (for GPU support)
- **Metal Framework** (macOS GPU support)
- **OpenCL** (cross-platform GPU support)

## Native Library Installation

The GGUF native bridge library (`libgguf_bridge`) can be loaded from multiple locations:

### Loading Order

1. **Explicit path** via system property `gollek.gguf.native.library.path`
2. **Extracted JAR resources** (bundled with the application)
3. **Standard installation** at `~/.gollek/libs/gguf_bridge/1.0.0/`
4. **Build directory** (development mode)
5. **System library path** via `java.library.path`

### Option 1: Automatic Installation via Maven

When building the project, the native library is automatically installed to the standard location:

```bash
cd gollek-gguf-converter
mvn clean install
```

This will:
- Build the native library from `gguf-bridge/` directory
- Copy it to `~/.gollek/libs/gguf_bridge/1.0.0/`
- Generate a SHA-256 checksum file
- Create a symlink at `~/.gollek/libs/libgguf_bridge`
- Clear macOS quarantine attributes (if applicable)

### Option 2: Manual Installation

```bash
# Build the native library
cd gguf-bridge
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release

# Install to standard location
mkdir -p ~/.gollek/libs/gguf_bridge/1.0.0
cp build/libgguf_bridge.dylib ~/.gollek/libs/gguf_bridge/1.0.0/  # macOS
# or
cp build/libgguf_bridge.so ~/.gollek/libs/gguf_bridge/1.0.0/     # Linux

# Generate checksum (optional but recommended)
shasum -a 256 ~/.gollek/libs/gguf_bridge/1.0.0/libgguf_bridge.dylib > \
  ~/.gollek/libs/gguf_bridge/1.0.0/libgguf_bridge.dylib.sha256

# Clear macOS quarantine (if applicable)
xattr -dr com.apple.quarantine ~/.gollek/libs/gguf_bridge/1.0.0/
```

### Option 3: Explicit Path Configuration

Set the library path via system property or environment variable:

```bash
# Using system property
java -Dgollek.gguf.native.library.path=/path/to/libgguf_bridge.dylib ...

# Using environment variable
export GOLLEK_GGUF_NATIVE_LIB_PATH=/path/to/libgguf_bridge.dylib
java ...
```

### Option 4: Directory Configuration

Set the library directory (the loader will look for the platform-specific library name):

```bash
# Using system property
java -Dgollek.gguf.native.library.dir=/path/to/libs ...

# Using environment variable
export GOLLEK_GGUF_NATIVE_LIB_DIR=/path/to/libs
java ...
```

## Building

### 1. Clone llama.cpp

```bash
git submodule add https://github.com/ggerganov/llama.cpp.git gguf-bridge/llama.cpp
git submodule update --init --recursive
```

### 2. Build Native Library

```bash
cd gguf-bridge
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

With CUDA support:
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DGGUF_ENABLE_CUDA=ON
cmake --build . --config Release
```

### 3. Build Java Service

```bash
cd converter-service
mvn clean install
```

## Running

### Development Mode

```bash
mvn quarkus:dev
```

The service will start on `http://localhost:8082` with:
- Swagger UI: http://localhost:8082/swagger-ui
- Health check: http://localhost:8082/q/health
- Metrics: http://localhost:8082/metrics

### Production Mode

```bash
java --enable-preview \
     --enable-native-access=ALL-UNNAMED \
     -Djava.library.path=/path/to/native/libs \
     -Dgollek.converter.base=~/.gollek/conversions \
     -Dgollek.model.base=~/.gollek/models \
     -jar target/quarkus-app/quarkus-run.jar
```

### Docker

```bash
docker build -f src/main/docker/Dockerfile.jvm -t gguf-converter:latest .
docker run -p 8082:8082 \
           -v /models:/var/lib/inference/models \
           -e converter.storage.base-path=/var/lib/inference/models \
           -e GOLLEK_CONVERTER_BASE=/var/lib/inference/models \
           -e GOLLEK_MODEL_BASE=/var/lib/inference/models \
           gguf-converter:latest
```

## API Usage

### Convert Model (Async)

```bash
curl -X POST http://localhost:8082/v1/converter/gguf/convert \
  -H "Content-Type: application/json" \
  -H "X-API-Key: tenant-123" \
  -d '{
    "inputPath": "models/llama-2-7b",
    "outputPath": "conversions",
    "quantization": "Q4_K_M",
    "numThreads": 8,
    "overwriteExisting": false,
    "dryRun": false
  }'
```

### Dry Run (Resolve Paths Only)

```bash
curl -X POST http://localhost:8082/v1/converter/gguf/convert \
  -H "Content-Type: application/json" \
  -H "X-API-Key: community" \
  -d '{
    "inputPath": "~/models/llama-2-7b",
    "outputPath": "conversions",
    "quantization": "Q4_K_M",
    "dryRun": true
  }'
```

### Preview Conversion (Resolve Only Endpoint)

```bash
curl -X POST http://localhost:8082/v1/converter/gguf/convert/preview \
  -H "Content-Type: application/json" \
  -H "X-API-Key: community" \
  -d '{
    "inputPath": "~/models/llama-2-7b",
    "outputPath": "conversions",
    "quantization": "Q4_K_M"
  }'
```

Relative `inputPath` values are resolved against `GOLLEK_MODEL_BASE` (default
`~/.gollek/models`). Relative `outputPath` values are resolved against
`GOLLEK_CONVERTER_BASE` (default `~/.gollek/conversions`). If `outputPath` is a
directory, the converter writes `<model>-<quant>.gguf` into that directory.
Set `overwriteExisting=true` to replace existing output files.
Set `dryRun=true` to resolve input/output paths without running conversion.
Preview and dry-run responses include `derivedOutputName`, `inputBasePath`, and `outputBasePath`.

Example preview response:
```json
{
  "success": true,
  "dryRun": true,
  "inputPath": "/Users/you/.gollek/models/llama-2-7b",
  "outputPath": "/Users/you/.gollek/conversions/llama-2-7b-q4_k_m.gguf",
  "derivedOutputName": "llama-2-7b-q4_k_m.gguf",
  "inputBasePath": "/Users/you/.gollek/models",
  "outputBasePath": "/Users/you/.gollek/conversions"
}
```

Response:
```json
{
  "conversionId": 1,
  "success": true,
  "requestId": "tenant-123",
  "outputPath": "/var/lib/inference/models/tenant-123/conversions/llama-2-7b-q4.gguf",
  "outputSize": "3.83 GB",
  "duration": "2m 15s",
  "compressionRatio": 3.52,
  "inputInfo": {
    "modelType": "llama",
    "architecture": "LlamaForCausalLM",
    "parameterCount": "7.2B",
    "fileSize": "13.48 GB",
    "format": "PyTorch"
  }
}
```

### Convert with Progress Stream

```bash
curl -N -H "X-API-Key: tenant-123" \
     -H "Accept: text/event-stream" \
     -X POST http://localhost:8082/v1/converter/gguf/convert/stream \
     -d '{"inputPath": "models/llama-2-7b", "outputPath": "out.gguf", "quantization": "Q4_K_M"}'
```

SSE Stream:
```
data: {"type":"progress","conversionId":1,"progress":0.1,"progressPercent":10,"stage":"Loading model"}

data: {"type":"progress","conversionId":1,"progress":0.5,"progressPercent":50,"stage":"Converting layers"}

data: {"type":"progress","conversionId":1,"progress":1.0,"progressPercent":100,"stage":"Complete"}

data: {"conversionId":1,"success":true,...}
```

### Detect Model Format

```bash
curl "http://localhost:8082/v1/converter/gguf/detect-format?path=models/llama-2-7b" \
     -H "X-API-Key: tenant-123"
```

Response:
```json
{
  "path": "models/llama-2-7b",
  "format": "pylibtorch",
  "displayName": "PyTorch",
  "convertible": true
}
```

### Get Model Info

```bash
curl "http://localhost:8082/v1/converter/gguf/model-info?path=models/llama-2-7b" \
     -H "X-API-Key: tenant-123"
```

Response:
```json
{
  "modelType": "llama",
  "architecture": "LlamaForCausalLM",
  "parameterCount": "7.2B",
  "parameterCountRaw": 7240000000,
  "numLayers": 32,
  "hiddenSize": 4096,
  "vocabSize": 32000,
  "contextLength": 4096,
  "fileSize": "13.48 GB",
  "fileSizeBytes": 14474000000,
  "format": "PyTorch",
  "estimatedMemoryGb": 16.8
}
```

### List Quantization Types

```bash
curl http://localhost:8082/v1/converter/gguf/quantizations
```

### Recommend Quantization

```bash
curl "http://localhost:8082/v1/converter/gguf/recommend-quantization?modelSizeGb=13.5&prioritizeQuality=true"
```

Response:
```json
{
  "recommended": "q4_k_m",
  "name": "Q4_K_M",
  "description": "4-bit K-quant (medium)",
  "qualityLevel": "MEDIUM_HIGH",
  "compressionRatio": 8.5,
  "reason": "Recommended for 13.5 GB model, prioritizing quality"
}
```

### Cancel Conversion

```bash
curl -X POST http://localhost:8082/v1/converter/gguf/convert/123/cancel \
     -H "X-API-Key: tenant-123"
```

## Configuration

Key configuration options in `application.yml`:

```yaml
converter:
  storage:
    base-path: ~/.gollek/conversions
    tenant-quota-gb: 100
    
  conversion:
    default-threads: 0  # Auto-detect
    default-quantization: f16
    max-concurrent-per-tenant: 2
    timeout-minutes: 120
    
  native:
    library-path: /usr/lib/inference
    enable-cuda: false
    enable-metal: false
```

Environment variables:
- `converter.storage.base-path`: Base storage path
- `CONVERTER_TENANT_QUOTA`: Per-tenant quota in GB
- `CONVERTER_DEFAULT_THREADS`: Default thread count
- `NATIVE_LIBRARY_PATH`: Path to native libraries
- `ENABLE_CUDA`: Enable CUDA support
 - `GOLLEK_CONVERTER_BASE`: Base path for relative output paths
 - `GOLLEK_MODEL_BASE`: Base path for relative input paths

## Quantization Guide

| Type | Description | Quality | Compression | Use Case |
|------|-------------|---------|-------------|----------|
| F16 | 16-bit float | ⭐⭐⭐⭐⭐ | 2x | Production, maximum quality |
| Q8_0 | 8-bit quantization | ⭐⭐⭐⭐ | 4x | High quality, good compression |
| Q5_K_M | 5-bit K-quant | ⭐⭐⭐⭐ | 6.5x | Excellent balance |
| Q4_K_M | 4-bit K-quant | ⭐⭐⭐ | 8.5x | **Recommended** for most use cases |
| Q4_K_S | 4-bit K-quant small | ⭐⭐⭐ | 9x | Aggressive compression |
| Q3_K_M | 3-bit K-quant | ⭐⭐ | 11x | Small models, acceptable quality |
| Q2_K | 2-bit K-quant | ⭐ | 16x | Extreme compression |

## Multi-Tenancy

The service provides complete tenant isolation:

1. **Storage Isolation**: Each tenant has a separate directory
2. **Quota Management**: Per-tenant storage limits
3. **Concurrent Limits**: Max conversions per tenant
4. **Path Validation**: Prevents path traversal attacks

Tenant context is resolved via API key (`X-API-Key` or `Authorization: ApiKey <key>`) and validated by `RequestContextFilter`. For community/standalone mode, use API key `community`.

## Performance

Typical conversion performance (8-core CPU):

| Model Size | Quantization | Time | Output Size |
|------------|--------------|------|-------------|
| 7B params | Q4_K_M | 2-3 min | 3.8 GB |
| 13B params | Q4_K_M | 4-6 min | 7.2 GB |
| 70B params | Q4_K_M | 25-35 min | 38 GB |

GPU acceleration (CUDA) provides 3-5x speedup for large models.

## Monitoring

### Metrics

Prometheus metrics exposed at `/metrics`:

```
# Conversion metrics
conversion_total{tenant,quantization,status}
conversion_duration_seconds{tenant,quantization}
conversion_file_size_bytes{tenant,type}
conversion_compression_ratio{tenant,quantization}

# Storage metrics
storage_used_bytes{tenant}
storage_quota_bytes{tenant}
storage_available_bytes{tenant}

# System metrics
jvm_memory_used_bytes{area}
http_server_requests_seconds{uri,method,status}
```

### Health Checks

```bash
curl http://localhost:8082/q/health
```

## Error Handling

Common errors and solutions:

| Error | Cause | Solution |
|-------|-------|----------|
| `GGUF_ERROR_FILE_NOT_FOUND` | Input path invalid | Check file path and permissions |
| `GGUF_ERROR_INVALID_FORMAT` | Unsupported format | Verify model format compatibility |
| `GGUF_ERROR_OUT_OF_MEMORY` | Insufficient RAM | Reduce concurrent conversions or add memory |
| `GGUF_ERROR_INVALID_QUANTIZATION` | Invalid quant type | Use `/quantizations` endpoint for valid types |
| `Storage quota exceeded` | Tenant over quota | Increase quota or clean up old files |

## Security

- **Path validation**: Prevents directory traversal
- **Tenant isolation**: Complete storage separation
- **Resource limits**: Prevents DoS via excessive conversions
- **Input validation**: Validates all parameters
- **Audit logging**: All operations logged with tenant context

## Development

### Running Tests

```bash
mvn test
```

### Integration Tests

```bash
mvn verify -Pintegration-tests
```

### Native Image Build

```bash
mvn package -Pnative
```

## License

Proprietary - Enterprise AI Platform

## Support

For issues or questions:
- Internal wiki: https://wiki.internal/inference-engine
- Slack: #inference-engine
- Email: ai-platform@company.com
