# Multi-stage build for LLM Inference Server (CPU-only)

# Build stage
FROM golang:1.21-alpine AS builder

# Install build dependencies
RUN apk add --no-cache gcc g++ make

# Set working directory
WORKDIR /app

# Copy go mod files
COPY go.mod go.sum ./

# Download dependencies
RUN go mod download

# Copy source code
COPY . .

# Build the application
RUN CGO_ENABLED=1 CGO_CFLAGS="-O3" CGO_CXXFLAGS="-O3" \
    go build -ldflags="-s -w" -o bin/llm-server ./cmd/server

# Runtime stage
FROM alpine:3.18

# Install runtime dependencies
RUN apk add --no-cache ca-certificates tzdata

# Create non-root user
RUN adduser -D -s /bin/sh llmuser

# Set working directory
WORKDIR /app

# Copy binary from builder
COPY --from=builder /app/bin/llm-server /usr/local/bin/llm-server

# Copy default configuration
COPY configs/config.yaml /app/config.yaml

# Create directory for models
RUN mkdir -p /app/models && chown -R llmuser:llmuser /app

# Switch to non-root user
USER llmuser

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8080/health || exit 1

# Default command
CMD ["llm-server", "--config", "/app/config.yaml"]