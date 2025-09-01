#!/bin/bash
set -e

# CPU Build Script for LLM Inference Server
echo "=== Building LLM Inference Server (CPU-only) ==="

# Set CGO flags for optimized CPU build
export CGO_ENABLED=1
export CGO_CFLAGS="-O3"
export CGO_CXXFLAGS="-O3"
export CGO_LDFLAGS=""

# Create output directory
mkdir -p bin

echo "Building CPU-only version..."
go build -ldflags="-s -w -X main.BuildType=cpu" \
    -o bin/llm-server-cpu ./cmd/server

echo "=== Build completed successfully ==="
echo "Binary: bin/llm-server-cpu"
echo "To run:"
echo "  ./bin/llm-server-cpu --model path/to/model.bin"

# Show binary size
if command -v ls &> /dev/null; then
    echo ""
    echo "Binary size: $(ls -lh bin/llm-server-cpu | awk '{print $5}')"
fi