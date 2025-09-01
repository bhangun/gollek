# Multi-stage Dockerfile for LLaMA.cpp Inference Server
FROM nvidia/cuda:12.2-devel-ubuntu22.04 AS cuda-builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    pkg-config \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Go
RUN wget https://go.dev/dl/go1.21.5.linux-amd64.tar.gz && \
    tar -C /usr/local -xzf go1.21.5.linux-amd64.tar.gz && \
    rm go1.21.5.linux-amd64.tar.gz

ENV PATH="/usr/local/go/bin:${PATH}"

# Clone and build llama.cpp with CUDA support
WORKDIR /build
RUN git clone https://github.com/ggerganov/llama.cpp.git
WORKDIR /build/llama.cpp
RUN cmake -B build \
    -DLLAMA_CUDA=ON \
    -DLLAMA_CUDA_F16=ON \
    -DLLAMA_FLASH_ATTN=ON \
    -DLLAMA_BLAS=ON \
    -DLLAMA_BLAS_VENDOR=OpenBLAS \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON

RUN cmake --build build --config Release -j$(nproc)

# CPU-only builder stage
FROM ubuntu:22.04 AS cpu-builder

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    pkg-config \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Go
RUN wget https://go.dev/dl/go1.21.5.linux-amd64.tar.gz && \
    tar -C /usr/local -xzf go1.21.5.linux-amd64.tar.gz && \
    rm go1.21.5.linux-amd64.tar.gz

ENV PATH="/usr/local/go/bin:${PATH}"

# Clone and build llama.cpp with CPU optimizations
WORKDIR /build
RUN git clone https://github.com/ggerganov/llama.cpp.git
WORKDIR /build/llama.cpp
RUN cmake -B build \
    -DLLAMA_BLAS=ON \
    -DLLAMA_BLAS_VENDOR=OpenBLAS \
    -DLLAMA_AVX=ON \
    -DLLAMA_AVX2=ON \
    -DLLAMA_AVX512=ON \
    -DLLAMA_FMA=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=ON

RUN cmake --build build --config Release -j$(nproc)

# Final runtime stage
FROM ubuntu:22.04 AS runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libopenblas0 \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install CUDA runtime (optional, will be ignored if not needed)
RUN apt-get update && apt-get install -y \
    nvidia-cuda-toolkit \
    || true

# Create app user
RUN groupadd -r llama && useradd -r -g llama llama

# Create directories
RUN mkdir -p /app /app/models /app/config /app/logs
RUN chown -R llama:llama /app

# Copy built libraries (default to CPU, override with --target for CUDA)
ARG BUILD_TYPE=cpu
COPY --from=${BUILD_TYPE}-builder /build/llama.cpp/build/libllama.so /usr/local/lib/
COPY --from=${BUILD_TYPE}-builder /build/llama.cpp/build/libggml.so /usr/local/lib/
COPY --from=${BUILD_TYPE}-builder /build/llama.cpp/llama.h /usr/local/include/
COPY --from=${BUILD_TYPE}-builder /build/llama.cpp/ggml.h /usr/local/include/
RUN ldconfig

# Set working directory
WORKDIR /app

# Copy application files
COPY --chown=llama:llama . .

# Build the Go application
USER llama
ENV CGO_ENABLED=1
ENV GOOS=linux
RUN go mod download
RUN go build -o llama-inference-server -ldflags="-w -s" .

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Set runtime configuration
ENV GIN_MODE=release
ENV SERVER_HOST=0.0.0.0
ENV SERVER_PORT=8080

# Run the application
CMD ["./wayang-inference"]