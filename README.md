# Wayang Inference Server

A production-grade REST/gRPC API server built with Go and llama.cpp for high-performance local LLM inference with GPU acceleration support.

## Features

- **High Performance**: Built with Go for excellent concurrency and llama.cpp for fast inference
- **GPU Acceleration**: Support for NVIDIA CUDA and Apple Metal
- **Streaming Responses**: Real-time streaming via Server-Sent Events (SSE)
- **Production Ready**: Comprehensive logging, health checks, and observability
- **OpenAI Compatible**: Compatible API endpoints for easy integration
- **Concurrent Safe**: Worker pool architecture for handling multiple requests
- **Configurable**: Extensive configuration options via files, environment variables, or CLI flags

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   HTTP Client   │───▶│   Gin Router     │───▶│  Request Queue  │
│                 │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Response Stream │◀───│  Worker Pool     │───▶│ LLaMA Engine    │
│     (SSE)       │    │                  │    │   (llama.cpp)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Quick Start

### Prerequisites

- Go 1.21 or later
- GCC/G++ for CGO compilation
- For GPU support:
  - **NVIDIA**: CUDA Toolkit 11.0+
  - **Apple**: macOS with Metal support

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd github.com/bhangun/wayang-inference
```

2. **Download dependencies:**
```bash
go mod download
```

3. **Build the server:**

**CPU-only version:**
```bash
make build-cpu
# or
./scripts/build-cpu.sh
```

**NVIDIA GPU version:**
```bash
make build-cuda
# or
./scripts/build-gpu-cuda.sh
```

**Apple Metal version:**
```bash
make build-metal
# or
./scripts/build-gpu-metal.sh
```

### Running the Server

1. **Download a model:**
```bash
mkdir -p models
# Download a GGUF model, for example:
wget https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.q4_0.bin -O models/llama-2-7b-chat.q4_0.bin
```

2. **Start the server:**

**CPU-only:**
```bash
./bin/llm-server-cpu --model models/llama-2-7b-chat.q4_0.bin
```

**With GPU acceleration:**
```bash
# NVIDIA GPU (adjust layers based on VRAM)
./bin/llm-server-cuda --model models/llama-2-7b-chat.q4_0.bin --gpu-layers 32

# Apple Metal
./bin/llm-server-metal --model models/llama-2-7b-chat.q4_0.bin --gpu-layers 32
```

3. **Test the API:**
```bash
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is the meaning of life?",
    "max_tokens": 150,
    "temperature": 0.7
  }'
```

**Streaming request:**
```bash
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Tell me a story",
    "max_tokens": 200,
    "temperature": 0.7,
    "stream": true
  }'
```

## Configuration

### Command Line Options

```bash
llm-server [flags]

Flags:
  -c, --config string       configuration file path
  -g, --gpu-layers int      number of layers to offload to GPU (0 for CPU-only)
  -m, --model string        path to the model file
  -H, --host string         server host address (default "0.0.0.0")
  -p, --port int            server port (default 8080)
  -w, --workers int         number of worker threads (default 2)
      --context-size int    context size for the model (default 2048)
      --log-level string    log level (debug, info, warn, error) (default "info")
  -v, --verbose             enable verbose logging
      --help                help for llm-server
      --version             version for llm-server
```

### Configuration File

Copy `configs/config.example.yaml` to `configs/config.yaml` and customize:

```yaml
server:
  host: "0.0.0.0"
  port: 8080
  
llm:
  model_path: "models/your-model.bin"
  context_size: 2048
  gpu_layers: 32          # Adjust based on your GPU VRAM
  threads: 4
  worker_pool_size: 2
  
logs:
  level: "info"
  format: "json"
```

### Environment Variables

All configuration options can be overridden with environment variables using the `LLM_SERVER_` prefix:

```bash
export LLM_SERVER_LLM_MODEL_PATH="/path/to/model.bin"
export LLM_SERVER_LLM_GPU_LAYERS=32
export LLM_SERVER_SERVER_PORT=8080
```

## API Reference

### Endpoints

- `POST /v1/completions` - Text completion (OpenAI compatible)
- `POST /v1/chat/completions` - Chat completion (alias for completions)
- `GET /v1/models` - List available models
- `GET /health` - Health check
- `GET /v1/metrics` - Server metrics

### Request Format

```json
{
  "prompt": "Your prompt here",
  "max_tokens": 150,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 40,
  "repeat_penalty": 1.1,
  "stop": ["Human:", "AI:"],
  "stream": false
}
```

### Response Format

**Non-streaming:**
```json
{
  "id": "cmpl-123456",
  "object": "text_completion",
  "created": 1677649420,
  "model": "llama-2-7b-chat",
  "choices": [
    {
      "index": 0,
      "text": "Generated text here...",
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 50,
    "total_tokens": 60
  }
}
```

**Streaming (Server-Sent Events):**
```
data: {"id":"cmpl-123456","object":"text_completion.chunk","created":1677649420,"model":"llama-2-7b-chat","choices":[{"index":0,"text":"Hello"}]}

data: {"id":"cmpl-123456","object":"text_completion.chunk","created":1677649420,"model":"llama-2-7b-chat","choices":[{"index":0,"text":" world"}]}

data: [DONE]
```

## GPU Configuration Guide

### NVIDIA CUDA

**Requirements:**
- NVIDIA GPU with CUDA Compute Capability 3.5+
- CUDA Toolkit 11.0 or later
- Driver version 450.80.02 or later

**Optimal settings for different GPUs:**
- RTX 4090 (24GB): `--gpu-layers 35`
- RTX 3080 (10GB): `--gpu-layers 25`
- RTX 3060 (8GB): `--gpu-layers 20`

### Apple Metal

**Requirements:**
- Apple Silicon Mac (M1/M2/M3)
- macOS 11.0 or later

**Optimal settings:**
- M1/M2 (8GB): `--gpu-layers 25`
- M1/M2 Pro (16GB): `--gpu-layers 32`
- M1/M2 Max (32GB+): `--gpu-layers 35`

## Docker Deployment

**CPU-only:**
```bash
docker build -f build/Dockerfile -t llm-server:cpu .
docker run -p 8080:8080 -v $(pwd)/models:/app/models llm-server:cpu
```

**With GPU (NVIDIA):**
```bash
docker build -f build/Dockerfile.cuda -t llm-server:cuda .
docker run --gpus all -p 8080:8080 -v $(pwd)/models:/app/models llm-server:cuda
```

## Performance Tuning

### CPU Optimization
- Set `threads` to the number of physical CPU cores
- Use `worker_pool_size: 1` for single-user scenarios
- Enable `use_mmap: true` for faster model loading

### GPU Optimization
- Start with `gpu_layers` equal to the total number of layers in your model
- Reduce if you encounter VRAM errors
- Use `worker_pool_size: 1` when using GPU to avoid conflicts

### Memory Optimization
- Enable `use_mmap: true` to reduce memory usage
- Consider quantized models (Q4, Q5, Q8) for lower memory requirements
- Set appropriate `context_size` based on your use case

## Monitoring and Observability

### Health Checks
- `GET /health` - Overall system health
- `GET /ready` - Kubernetes readiness probe
- `GET /live` - Kubernetes liveness probe

### Metrics
- `GET /v1/metrics` - Detailed performance metrics
- Structured JSON logging
- Request/response tracking
- Resource utilization monitoring

## Development

### Building from Source
```bash
# Development build with race detection
make build-dev

# Run tests
make test

# Run with race detection
make test-race

# Format code
make fmt

# Lint code
make lint
```

### Project Structure
```
github.com/bhangun/wayang-inference/
├── cmd/server/          # Application entry point
├── internal/            # Private application code
│   ├── config/          # Configuration management
│   ├── llm/             # LLM engine and worker pool
│   ├── server/          # HTTP server and handlers
│   ├── streaming/       # Server-Sent Events implementation
│   └── health/          # Health check implementation
├── pkg/                 # Public packages
│   ├── types/           # API types and structures
│   └── logger/          # Structured logging
├── configs/             # Configuration files
├── scripts/             # Build and deployment scripts
└── build/               # Build configurations and Dockerfiles
```

## Troubleshooting

### Common Issues

**"Model file not found"**
- Ensure the model path is correct
- Check file permissions
- Verify the model file is a valid GGUF/GGML format

**"CUDA out of memory"**
- Reduce `gpu_layers` parameter
- Use a smaller model or quantized version
- Ensure no other processes are using GPU memory

**"Failed to load CUDA libraries"**
- Install CUDA Toolkit
- Add CUDA libraries to `LD_LIBRARY_PATH`
- Verify CUDA installation with `nvcc --version`

**High memory usage**
- Enable `use_mmap: true`
- Reduce `context_size`
- Use quantized models (Q4, Q8)

**Slow inference speed**
- Increase `gpu_layers` if using GPU
- Optimize `threads` setting for CPU
- Consider using a smaller model

### Debug Logging
Enable debug logging to troubleshoot issues:
```bash
llm-server --log-level debug --verbose
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

[Your License Here]

## Support

- Create an issue for bug reports or feature requests
- Check the troubleshooting section above
- Review the configuration examples