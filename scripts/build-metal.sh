#!/bin/bash
set -e

# Metal Build Script for LLM Inference Server (macOS)
echo "=== Building LLM Inference Server with Metal Support ==="

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "Error: Metal GPU acceleration is only available on macOS."
    exit 1
fi

# Check for Xcode command line tools
if ! command -v xcrun &> /dev/null; then
    echo "Error: Xcode command line tools not found."
    echo "Please install with: xcode-select --install"
    exit 1
fi

# Check for Metal framework
if ! xcrun -find metal &> /dev/null; then
    echo "Error: Metal compiler not found."
    echo "Please ensure Xcode or command line tools are properly installed."
    exit 1
fi

echo "Metal GPU acceleration support detected"

# Set CGO flags for Metal
export CGO_ENABLED=1
export CGO_CFLAGS="-O3 -DGGML_USE_METAL"
export CGO_CXXFLAGS="-O3 -DGGML_USE_METAL"
export CGO_LDFLAGS="-framework Metal -framework Foundation -framework MetalKit"

# Create output directory
mkdir -p bin

echo "Building with Metal support..."
go build -ldflags="-s -w -X main.BuildType=metal" \
    -o bin/llm-server-metal ./cmd/server

echo "=== Build completed successfully ==="
echo "Binary: bin/llm-server-metal"
echo "To run with GPU acceleration on Apple Silicon:"
echo "  ./bin/llm-server-metal --model path/to/model.bin --gpu-layers 32"

# Verify Metal frameworks are linked
echo ""
echo "Verifying Metal framework linking..."
if otool -L bin/llm-server-metal | grep -q Metal; then
    echo "Metal frameworks successfully linked:"
    otool -L bin/llm-server-metal | grep Metal
else
    echo "Warning: Metal frameworks may not be properly linked."
fi