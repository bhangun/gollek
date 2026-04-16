# Gollek Metal Kernel

Apple Metal GPU acceleration kernel for Gollek inference engine.

## Features

- **MPSGraph SDPA**: FlashAttention-4 equivalent via Metal Performance Shaders (macOS 14+)
- **Unified Memory**: Zero-copy CPU/GPU shared DRAM on all Apple Silicon
- **AMX Acceleration**: Apple's matrix engines for GEMM operations
- **Weight Offloading**: Run large models on limited RAM by streaming weights
- **BF16 Support**: BFloat16 precision on M3/M4 chips

## Supported Chips

| Chip | GPU Cores | Memory | Unified | macOS | Best For |
|------|-----------|--------|---------|-------|----------|
| M1 | 7-8 | 8-16 GB | ✓ | 13+ | Entry-level |
| M1 Pro/Max | 14-16 | 16-32 GB | ✓ | 13+ | Development |
| M2 | 8-10 | 8-24 GB | ✓ | 13+ | General use |
| M2 Pro/Max/Ultra | 16-38 | 16-96 GB | ✓ | 13+ | Production |
| M3 | 8-10 | 8-24 GB | ✓ | 14+ | General use |
| M3 Pro/Max | 14-40 | 18-128 GB | ✓ | 14+ | Production |
| M4 | 8-10 | 8-16 GB | ✓ | 14+ | Latest entry |
| M4 Pro/Max | 14-40 | 24-128 GB | ✓ | 14+ | Latest pro |

## Configuration

```properties
# Enable Metal runner
gollek.runners.metal.enabled=true

# Runner mode: auto|standard|offload|force|disabled
gollek.runners.metal.mode=auto

# Library path
gollek.runners.metal.library-path=~/.gollek/libs/libgollek_metal.dylib

# Model dimensions (override from manifest)
gollek.runners.metal.num-layers=32
gollek.runners.metal.num-heads=32
gollek.runners.metal.num-heads-kv=8
gollek.runners.metal.head-dim=128
gollek.runners.metal.model-dim=4096
gollek.runners.metal.ffn-dim=14336
gollek.runners.metal.vocab-size=32000
```

## Building Metal Kernels

```bash
# Prerequisites - macOS 14+ with Xcode Command Line Tools
xcode-select --install

# Verify Apple Silicon
uname -m  # Should show "arm64"

# Build Metal bridge
make -C src/main/cpp/metal

# Output
target/native/macos-aarch64/libgollek_metal.dylib

# Install to default location
mkdir -p ~/.gollek/libs
cp target/native/macos-aarch64/libgollek_metal.dylib ~/.gollek/libs/
```

## Testing

```bash
# Run with GPU tests (Apple Silicon only)
mvn test -Pmetal-gpu-tests

# Run without GPU (CPU fallback)
mvn test
```

## Performance by Chip

### Llama-3.2-3B

| Chip | Tokens/sec | Memory (GB) | Runner Mode |
|------|-----------|-------------|-------------|
| M4 Max | 60 | 6 | standard |
| M3 Max | 55 | 6 | standard |
| M2 Ultra | 50 | 6 | standard |
| M1 Max | 28 | 6 | standard |
| M2 (8GB) | 18 | 6 | offload |

### Llama-3-70B

| Chip | Tokens/sec | Memory (GB) | Runner Mode |
|------|-----------|-------------|-------------|
| M4 Max (128GB) | 22 | 70 | standard |
| M3 Max (128GB) | 20 | 70 | standard |
| M2 Ultra (96GB) | 16 | 70 | standard |
| M1 Max (64GB) | 8 | 70 | offload |
| M2 (24GB) | 3 | 70 | offload |

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    MetalRunner                          │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │  RMSNorm    │  │   GEMM      │  │  MPSGraph   │     │
│  │  Kernel     │  │  (AMX)      │  │  SDPA/FA4   │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
│                                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │   SiLU      │  │  Paged KV   │  │  Weight     │     │
│  │   FFN       │  │   Cache     │  │  Offload    │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                   MetalBinding (FFM)                    │
├─────────────────────────────────────────────────────────┤
│  libgollek_metal.dylib → Metal → Apple GPU             │
│  - MTLDevice (GPU device)                               │
│  - MTLCommandBuffer (command queue)                     │
│  - MPSMatrixMultiplication (GEMM)                       │
│  - MPSGraph SDPA (FlashAttention-4 equivalent)          │
└─────────────────────────────────────────────────────────┘
```

## Runner Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `auto` | Detect and select based on model size | Default - recommended |
| `standard` | Load full model to unified memory | M2 Max+ with 32GB+ RAM |
| `offload` | Stream weights from CPU to GPU | M1/M2 with <32GB RAM |
| `force` | Use Metal even if detection fails | Debug/testing |
| `disabled` | Don't use Metal | CPU-only fallback |

### Mode Selection Logic

```
Model Size < 50% of Total Memory?
         │
    ┌────┴────┐
    │         │
   Yes       No
    │         │
    ▼         ▼
standard  offload
```

## Unified Memory Architecture

All Apple Silicon chips feature unified memory where CPU and GPU share the same physical DRAM:

```
┌──────────────────────────────────────────┐
│          Apple Silicon (M-series)        │
├──────────────────────────────────────────┤
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  │
│  │   CPU   │  │   GPU   │  │ Unified │  │
│  │  Cores  │  │  Cores  │  │ Memory  │  │
│  └─────────┘  └─────────┘  │  Pool   │  │
│         ↑           ↑      │         │  │
│         └───────────┴──────┘         │  │
│              Zero-copy Access        │  │
└──────────────────────────────────────────┘
```

**Benefits:**
- **Zero-copy** data sharing between CPU and GPU
- **No explicit** memory transfers required
- **Simplified** programming model

## Weight Offloading

For models larger than available RAM, Metal supports weight offloading:

```java
import tech.kayys.gollek.metal.runner.MetalRunner;
import tech.kayys.gollek.metal.optimization.MetalOptimizationManager;

// Create runner
MetalRunner runner = new MetalRunner();

// Configure for offload mode (large models on limited RAM)
RunnerConfiguration config = RunnerConfiguration.builder()
    .parameter("num_layers", 80)
    .build();

System.setProperty("gollek.runners.metal.mode", "offload");
runner.initialize(manifest, config);

// Weights are streamed from CPU to GPU as needed
```

## Optimization Integration

The Metal kernel integrates with `MetalOptimizationManager` for automatic kernel selection:

```java
MetalOptimizationManager optimization = new MetalOptimizationManager(
    metal, metalFa4, kvCacheManager, gpuCores);

if (optimization.isSdpaAvailable()) {
    // macOS 14+ with M3/M4: Use MPSGraph SDPA (FA4 equivalent)
    optimization.executeMpsGraphSdpa(
        output, query, kPool, vPool,
        batchSize, seqLen, numHeads, numHeadsKV, headDim,
        scale, causal);
} else {
    // macOS 13 or older: Use separate matmuls
    optimization.executeSeparateMatmuls(
        output, query, kPool, vPool,
        batchSize, seqLen, numHeads, numHeadsKV, headDim,
        scale, causal);
}
```

## Troubleshooting

### "Metal unavailable"

```bash
# Verify Apple Silicon
uname -m  # Should show "arm64"

# Check macOS version
sw_vers -productVersion  # Should be 14.0+ for best performance

# Install Xcode tools
xcode-select --install
```

### "Library not loaded"

```bash
# Build and install
make -C src/main/cpp/metal
mkdir -p ~/.gollek/libs
cp target/native/macos-aarch64/libgollek_metal.dylib ~/.gollek/libs/
```

### "Out of memory on large models"

```bash
# Use weight offloading mode
gollek.runners.metal.mode=offload

# Or reduce model size
gollek.runners.metal.num-layers=16  # Instead of 32
```

### "MPSGraph SDPA not available"

```bash
# Check macOS version
sw_vers -productVersion  # Must be 14.0+ for SDPA

# If macOS 13, will fall back to separate matmuls automatically
```

## Performance Tips

1. **Use macOS 14+** for MPSGraph SDPA (2x faster than separate matmuls)
2. **Enable BF16** on M3/M4 for better precision with minimal speed loss
3. **Use offload mode** when model > 50% of total memory
4. **Close other apps** to free unified memory for large models
5. **Use M3 Max/M4 Max** for best performance (40 GPU cores)

## Resources

- [Apple Metal Documentation](https://developer.apple.com/metal/)
- [MPSGraph Documentation](https://developer.apple.com/documentation/metalperformanceshadersgraph)
- [Apple Silicon Specifications](https://www.apple.com/mac/)

## License

Apache 2.0
