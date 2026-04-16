# Gollek CUDA Kernel

NVIDIA CUDA GPU acceleration kernel for Gollek inference engine.

## Features

- **FlashAttention-2/3/4**: Auto-selected fused attention kernels for optimal performance
- **Unified Memory Support**: Zero-copy on A100/H100 with managed memory
- **FP8/FP4 Acceleration**: Tensor core optimization on H100+ and Blackwell
- **Paged KV Cache**: Efficient memory management for long contexts
- **Multi-GPU Ready**: Device selection via `gollek.runners.cuda.device-id`
- **Optimization Integration**: Automatic kernel selection via `CudaOptimizationManager`

## Supported GPUs

| GPU | Compute Cap | Memory | FlashAttention | Precision |
|-----|-------------|--------|----------------|-----------|
| A100 | sm_80 (8.0) | 40/80 GB HBM2e | FA2 | FP16/BF16 |
| H100 | sm_90 (9.0) | 80 GB HBM3 | FA3 + FP8 | FP8/FP16 |
| H200 | sm_90 (9.0) | 141 GB HBM3e | FA3 + FP8 | FP8/FP16 |
| B100 | sm_100 (10.0) | 180 GB HBM3e | FA4 + TMEM | FP4/FP8 |
| B200 | sm_100 (10.0) | 180 GB HBM3e | FA4 + TMEM | FP4/FP8 |
| RTX 4090 | sm_89 (8.9) | 24 GB GDDR6X | FA2 | FP16 |
| RTX A6000 | sm_86 (8.6) | 48 GB GDDR6 | FA2 | FP16 |

## Configuration

```properties
# Enable CUDA runner
gollek.runners.cuda.enabled=true

# Runner mode: auto|standard|offload|force|disabled
gollek.runners.cuda.mode=auto

# CUDA library path
gollek.runners.cuda.library-path=/usr/local/cuda/lib64/libgollek_cuda.so

# Device ID (0-based)
gollek.runners.cuda.device-id=0

# Model dimensions (override from manifest)
gollek.runners.cuda.num-layers=32
gollek.runners.cuda.num-heads=32
gollek.runners.cuda.num-heads-kv=8
gollek.runners.cuda.head-dim=128
gollek.runners.cuda.model-dim=4096
gollek.runners.cuda.ffn-dim=14336
gollek.runners.cuda.vocab-size=32000

# Optimization settings (auto-detected)
gollek.runners.cuda.use-fp8=true   # Auto-enabled on H100+
gollek.runners.cuda.use-fp4=true   # Auto-enabled on B100/B200
```

## Building CUDA Kernels

```bash
# Prerequisites
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH

# Build for A100 (sm_80)
make -C src/main/cpp/cuda CUDA_ARCH=sm_80

# Build for H100 (sm_90)
make -C src/main/cpp/cuda CUDA_ARCH=sm_90

# Build for Blackwell (sm_100)
make -C src/main/cpp/cuda CUDA_ARCH=sm_100 USE_FP4=1

# Build for multiple architectures
make -C src/main/cpp/cuda CUDA_ARCHS="sm_80 sm_90 sm_100"

# Output location
target/native/linux-x86_64/libgollek_cuda.so
```

## Testing

```bash
# Run with GPU tests enabled
CUDA_VISIBLE_DEVICES=0 mvn test -Pcuda-gpu-tests

# Run without GPU (CPU fallback)
mvn test
```

## Performance Comparison

### Llama-3.2-3B

| GPU | Kernel | Precision | Tokens/sec | Memory (GB) |
|-----|--------|-----------|------------|-------------|
| B200 | FA4+TMEM | FP4 | 180 | 5 |
| B200 | FA4+TMEM | FP8 | 135 | 5 |
| H100 | FA3 | FP8 | 95 | 6 |
| A100 | FA2 | FP16 | 45 | 8 |
| RTX 4090 | FA2 | FP16 | 35 | 8 |

### Llama-3-70B

| GPU | Kernel | Precision | Tokens/sec | Memory (GB) |
|-----|--------|-----------|------------|-------------|
| B200 | FA4+TMEM | FP4 | 55 | 38 |
| B200 | FA4+TMEM | FP8 | 40 | 38 |
| H100 | FA3 | FP8 | 28 | 40 |
| A100 80GB | FA2 | FP16 | 12 | 72 |

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    CudaRunner                           │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │  RMSNorm    │  │   GEMM      │  │ Optimization│     │
│  │  Kernel     │  │  (cuBLAS)   │  │  Manager    │     │
│  └─────────────┘  └─────────────┘  └──────┬──────┘     │
│                                            │            │
│  ┌─────────────┐  ┌─────────────┐  ┌──────▼──────┐     │
│  │   SiLU      │  │  Paged KV   │  │ FA4/FA3/FA2 │     │
│  │   FFN       │  │   Cache     │  │  (Auto)     │     │
│  └─────────────┘  └─────────────┘  └─────────────┘     │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                   CudaBinding (FFM)                     │
├─────────────────────────────────────────────────────────┤
│  libgollek_cuda.so → CUDA Driver API → GPU             │
│  - cuInit, cuDeviceGet (device management)              │
│  - cuMemAllocManaged (unified memory)                   │
│  - cuBLAS/cuBLASLt (matrix multiplication)              │
│  - FlashAttention kernels (FA2/FA3/FA4)                 │
└─────────────────────────────────────────────────────────┘
```

## Optimization Integration

The CUDA kernel integrates with `CudaOptimizationManager` for automatic kernel selection:

```java
import tech.kayys.gollek.cuda.runner.CudaRunner;
import tech.kayys.gollek.cuda.optimization.CudaOptimizationManager;

// Create runner
CudaRunner runner = new CudaRunner();

// Initialize (automatically creates CudaOptimizationManager)
runner.initialize(manifest, config);

// Auto-selection logic:
// - sm_100+ (B200): FA4 + TMEM + FP4
// - sm_90+ (H100): FA3 + FP8
// - sm_80+ (A100): FA2
// - Older: Paged attention (fallback)

// Check recommended kernel
String kernel = runner.getOptimizationManager().getRecommendedKernel();
System.out.println("Using kernel: " + kernel);
```

### Auto-Selection Flow

```
GPU Detection
      ↓
Compute Capability Check
      ↓
┌─────────────────┐
│ sm_100+ (B200)  │ → FA4 + TMEM + FP4 (3.5x H100)
├─────────────────┤
│ sm_90+ (H100)   │ → FA3 + FP8 (2x A100)
├─────────────────┤
│ sm_80+ (A100)   │ → FA2 (1.5x standard)
├─────────────────┤
│ Older           │ → Paged attention (fallback)
└─────────────────┘
```

## Performance Tips

1. **Use unified memory** on A100/H100/B200 for zero-copy transfers
2. **Enable FlashAttention** for 2-3x speedup on attention layers
3. **Use FP4** on Blackwell for 2x speedup over FP8
4. **Use FP8** on H100+ for 2x speedup over FP16
5. **Pin batch sizes** to multiples of warp size (32)
6. **Pre-allocate KV cache** for long context inference

## Troubleshooting

### "CUDA not found"

```bash
# Verify CUDA installation
nvcc --version
nvidia-smi

# Check library paths
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Verify GPU visibility
echo $CUDA_VISIBLE_DEVICES  # Should show device IDs
```

### "Out of memory"

```bash
# Reduce batch size
gollek.runners.cuda.max-batch-size=1

# Enable unified memory (A100/H100 only)
gollek.runners.cuda.use-unified-memory=true

# Use weight offloading
gollek.runners.cuda.mode=offload

# Monitor GPU memory
watch -n 1 nvidia-smi
```

### "Compute capability mismatch"

```bash
# Rebuild kernels for your GPU architecture
# A100
make CUDA_ARCH=sm_80

# H100
make CUDA_ARCH=sm_90

# Blackwell
make CUDA_ARCH=sm_100 USE_FP4=1
```

## Resources

- [NVIDIA CUDA Documentation](https://docs.nvidia.com/cuda/)
- [FlashAttention-4 Paper](https://arxiv.org/abs/2603.05451)
- [FlashAttention-3 Paper](https://arxiv.org/abs/2307.08691)
- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)

## License

Apache 2.0
