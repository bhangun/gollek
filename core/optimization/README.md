# Gollek Optimization Plugins

GPU kernel optimization plugins for high-performance AI inference.

## Overview

The Gollek Optimization Plugin system provides a modular architecture for GPU kernel optimizations, allowing you to:

- **Hot-swap kernel implementations** without recompiling
- **Auto-detect hardware** and select optimal kernels
- **Mix and match** different optimization strategies
- **Develop custom kernels** as standalone plugins

## Available Plugins

| Plugin | Description | GPU Requirement | Speedup |
|--------|-------------|-----------------|---------|
| **FlashAttention-3** | FA3 kernel for Hopper+ | H100+ (CC 9.0+) | Up to 3x |
| **FlashAttention-4** | FA4 kernel for Blackwell | B200+ (CC 10.0+) | Up to 5x |
| **PagedAttention** | vLLM-style paged KV cache | All CUDA GPUs | 2-4x |
| **KV Cache** | Optimized KV cache management | All GPUs | 1.5-2x |
| **Prompt Cache** | Prompt caching for repeated queries | All GPUs | 5-10x* |
| **Prefill-Decode** | Separate prefill/decode kernels | All GPUs | 1.3-1.8x |
| **Hybrid Attention** | Dynamic attention strategy | All GPUs | 1.2-1.5x |
| **Elastic EP** | Elastic expert parallelism | Multi-GPU | 2-3x* |
| **PerfMode** | Performance mode tuning | All GPUs | 1.1-1.3x |
| **QLoRA** | Quantized LoRA inference | All GPUs | 2-3x* |
| **Weight Offload** | CPU/GPU weight offloading | Limited VRAM | Enables larger models |
| **Eviction Compression** | KV cache eviction strategies | All GPUs | 1.5-2x |
| **Wait Scheduler** | Async execution scheduling | All GPUs | 1.2-1.8x |

*For specific workloads

## Quick Start

### Installation

#### Option 1: Build from Source

```bash
cd inference-gollek/plugins/gollek-plugin-optimization
mvn clean install -Pinstall-plugin
```

This installs all optimization plugins to `~/.gollek/plugins/optimization/`

#### Option 2: Download Pre-built JARs

```bash
# Download specific plugin
wget https://github.com/gollek-ai/gollek/releases/download/v1.0.0/gollek-plugin-fa3-1.0.0.jar

# Copy to optimization plugins directory
cp gollek-plugin-fa3-1.0.0.jar ~/.gollek/plugins/optimization/
```

### Configuration

Create `~/.gollek/plugins/optimization/optimization-config.json`:

```json
{
  "flash-attention-3": {
    "enabled": true,
    "tile_size": 128,
    "use_tensor_cores": true
  },
  "paged-attention": {
    "enabled": true,
    "block_size": 16,
    "num_blocks": 1024
  },
  "prompt-cache": {
    "enabled": true,
    "max_cache_size_gb": 8,
    "ttl_seconds": 3600
  }
}
```

### Usage Example

```java
import tech.kayys.gollek.plugin.optimization.OptimizationPluginManager;
import tech.kayys.gollek.plugin.optimization.ExecutionContext;

// Get plugin manager
OptimizationPluginManager manager = OptimizationPluginManager.getInstance();

// Register plugins (auto-discovered from classpath or plugin directory)
manager.register(new FlashAttention3Plugin());
manager.register(new PagedAttentionPlugin());

// Initialize with configuration
Map<String, Object> config = loadConfig();
manager.initialize(config);

// During inference, apply optimizations
ExecutionContext context = createExecutionContext(...);
List<String> applied = manager.applyOptimizations(context);

System.out.println("Applied optimizations: " + applied);
```

## Plugin Development

### Creating a Custom Optimization Plugin

#### 1. Implement the OptimizationPlugin Interface

```java
package com.example.plugin;

import tech.kayys.gollek.plugin.optimization.*;
import java.util.*;

public class MyCustomKernelPlugin implements OptimizationPlugin {
    
    @Override
    public String id() {
        return "my-custom-kernel";
    }
    
    @Override
    public String name() {
        return "My Custom Kernel";
    }
    
    @Override
    public String description() {
        return "Custom optimization for specialized workloads";
    }
    
    @Override
    public boolean isAvailable() {
        // Check hardware requirements
        return GpuDetector.isNvidia() && 
               GpuDetector.getComputeCapability() >= 80;
    }
    
    @Override
    public boolean apply(ExecutionContext context) {
        // Get model parameters
        int hiddenSize = context.getParameter("hidden_size", 4096);
        int numHeads = context.getParameter("num_heads", 32);
        
        // Get memory buffers
        MemoryBuffer qBuffer = context.getBuffer("q").orElse(null);
        MemoryBuffer kBuffer = context.getBuffer("k").orElse(null);
        MemoryBuffer vBuffer = context.getBuffer("v").orElse(null);
        
        if (qBuffer == null || kBuffer == null || vBuffer == null) {
            return false;
        }
        
        // Apply your custom kernel
        return applyCustomKernel(
            qBuffer.getPointer(),
            kBuffer.getPointer(),
            vBuffer.getPointer(),
            context.getCudaStream()
        );
    }
    
    // Native method for your custom kernel
    private native boolean applyCustomKernel(
        long qPtr, long kPtr, long vPtr, long stream
    );
}
```

#### 2. Create pom.xml

```xml
<project>
    <groupId>com.example</groupId>
    <artifactId>gollek-plugin-my-custom</artifactId>
    <version>1.0.0</version>
    
    <dependencies>
        <dependency>
            <groupId>tech.kayys.gollek</groupId>
            <artifactId>gollek-plugin-optimization-core</artifactId>
            <version>1.0.0-SNAPSHOT</version>
        </dependency>
        <dependency>
            <groupId>net.java.dev.jna</groupId>
            <artifactId>jna</artifactId>
            <version>5.14.0</version>
        </dependency>
    </dependencies>
    
    <build>
        <plugins>
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-jar-plugin</artifactId>
                <configuration>
                    <archive>
                        <manifestEntries>
                            <Plugin-Id>my-custom-kernel</Plugin-Id>
                            <Plugin-Class>com.example.plugin.MyCustomKernelPlugin</Plugin-Class>
                            <Plugin-Type>optimization</Plugin-Type>
                        </manifestEntries>
                    </archive>
                </configuration>
            </plugin>
        </plugins>
    </build>
</project>
```

#### 3. Build and Deploy

```bash
mvn clean package
cp target/gollek-plugin-my-custom-1.0.0.jar ~/.gollek/plugins/optimization/
```

### Plugin Manifest Entries

| Entry | Required | Description |
|-------|----------|-------------|
| `Plugin-Id` | Yes | Unique plugin identifier |
| `Plugin-Class` | Yes | Main plugin class |
| `Plugin-Type` | Yes | Must be `optimization` |
| `Plugin-Dependencies` | No | Comma-separated dependency IDs |
| `GPU-Arch` | No | Supported GPU architectures |
| `CUDA-Compute-Capability` | No | Minimum CUDA compute capability |

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Inference Engine                           │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│         OptimizationPluginManager                       │
│  - Plugin discovery                                     │
│  - Lifecycle management                                 │
│  - Hardware detection                                   │
│  - Priority-based execution                             │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│          Optimization Plugins                           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │   FA3    │ │   FA4    │ │  Paged   │ │   KV     │  │
│  │ Plugin   │ │ Plugin   │ │ Attention│ │  Cache   │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│  │  Prompt  │ │ Prefill  │ │  Hybrid  │ │  QLoRA   │  │
│  │  Cache   │ │  Decode  │ │  Attn    │ │          │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘  │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│           Native Kernel Libraries                       │
│  libgollek_fa3.so | libgollek_fa4.so | libcustom.so     │
└─────────────────────────────────────────────────────────┘
```

## Hardware Compatibility

### NVIDIA GPUs

| GPU Series | Architecture | Compute Capability | Supported Plugins |
|------------|--------------|-------------------|-------------------|
| H100, H200 | Hopper | 9.0 | All (FA3, FA4, Paged, etc.) |
| B200, GB200 | Blackwell | 10.0 | All (optimized for FA4) |
| A100, A800 | Ampere | 8.0 | FA2, Paged, KV Cache, etc. |
| A40, RTX A6000 | Ampere | 8.6 | FA2, Paged, KV Cache, etc. |
| RTX 4090, 4080 | Ada Lovelace | 8.9 | FA2, Paged, KV Cache, etc. |
| RTX 3090, 3080 | Ampere | 8.6 | FA2, Paged, KV Cache, etc. |
| V100 | Volta | 7.0 | Paged, KV Cache (limited) |
| T4 | Turing | 7.5 | Paged, KV Cache (limited) |

### AMD GPUs

| GPU Series | Architecture | Supported Plugins |
|------------|--------------|-------------------|
| MI300X, MI300A | CDNA 3 | Paged, KV Cache, Prompt Cache |
| MI250X, MI250 | CDNA 2 | Paged, KV Cache |
| MI210, MI200 | CDNA 2 | Paged, KV Cache |
| RX 7900 XTX | RDNA 3 | Limited support |

## Performance Tuning

### FlashAttention-3

```json
{
  "flash-attention-3": {
    "tile_size": 128,        // 64, 128, 256 (larger = more memory, faster)
    "use_tensor_cores": true, // Use Tensor Cores (Hopper+)
    "causal": true,          // Causal attention mask
    "softmax_scale": 1.0     // Softmax scaling factor
  }
}
```

### PagedAttention

```json
{
  "paged-attention": {
    "block_size": 16,         // 8, 16, 32 (smaller = finer granularity)
    "num_blocks": 1024,       // Number of KV cache blocks
    "max_context": 32768      // Maximum context length
  }
}
```

### Prompt Cache

```json
{
  "prompt-cache": {
    "max_cache_size_gb": 8,   // Maximum cache size in GB
    "ttl_seconds": 3600,      // Cache entry TTL
    "min_prompt_len": 256,    // Minimum prompt length to cache
    "hash_algorithm": "md5"   // md5, sha256
  }
}
```

## Troubleshooting

### Plugin Not Loading

```bash
# Check if plugin JAR is in correct directory
ls -la ~/.gollek/plugins/optimization/*.jar

# Check plugin manifest
unzip -p gollek-plugin-fa3.jar META-INF/MANIFEST.MF

# Check logs for errors
tail -f ~/.gollek/logs/gollek.log | grep -i optimization
```

### Native Library Not Found

```bash
# Ensure native library is in library path
export LD_LIBRARY_PATH=/path/to/native/libs:$LD_LIBRARY_PATH

# Or copy to system library directory
sudo cp libgollek_fa3.so /usr/local/lib/
sudo ldconfig
```

### Performance Issues

```bash
# Check which optimizations are active
curl http://localhost:8080/api/v1/optimizations/status

# Benchmark with/without specific plugin
gollek-cli benchmark --model llama-3-8b --no-optimizations
gollek-cli benchmark --model llama-3-8b --optimizations fa3,paged
```

## Resources

- [Plugin Developer Guide](PLUGIN_DEVELOPER_GUIDE.md)
- [FlashAttention-3 Paper](https://arxiv.org/abs/2307.08691)
- [PagedAttention Paper](https://arxiv.org/abs/2309.06180)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/)

## License

MIT License - See LICENSE file for details.
