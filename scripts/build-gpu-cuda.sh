#!/bin/bash
set -e

# CUDA Build Script for LLM Inference Server
echo "=== Building LLM Inference Server with CUDA Support ==="

# Check if CUDA is available
if ! command -v nvcc &> /dev/null; then
    echo "Error: CUDA compiler (nvcc) not found."
    echo "Please install CUDA toolkit from: https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

# Check for CUDA libraries
if ! ldconfig -p | grep -q libcublas; then
    echo "Error: CUDA libraries not found."
    echo "Please ensure CUDA runtime is properly installed and LD_LIBRARY_PATH is configured."
    exit 1
fi

# Get CUDA version
CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -d',' -f1 | cut -d'V' -f2)
echo "Found CUDA version: $CUDA_VERSION"

# Set CGO flags for CUDA
export CGO_ENABLED=1
export CGO_CFLAGS="-O3 -DGGML_USE_CUBLAS"
export CGO_CXXFLAGS="-O3 -DGGML_USE_CUBLAS"
export CGO_LDFLAGS="-lcublas -lcudart -lcublasLt"

# Create output directory
mkdir -p bin

echo "Building with CUDA support..."
go build -ldflags="-s -w -X main.BuildType=cuda -X main.CUDAVersion=$CUDA_VERSION" \
    -o bin/llm-server-cuda ./cmd/server

echo "=== Build completed successfully ==="
echo "Binary: bin/llm-server-cuda"
echo "To run with GPU acceleration, use the --gpu-layers flag:"
echo "  ./bin/llm-server-cuda --model path/to/model.bin --gpu-layers 32"

# Verify CUDA libraries can be loaded
echo ""
echo "Testing CUDA library loading..."
if ldd bin/llm-server-cuda | grep -q "not found"; then
    echo "Warning: Some CUDA libraries may not be found. Please check your CUDA installation."
    ldd bin/llm-server-cuda | grep "not found"
else
    echo "CUDA libraries successfully linked."
fi