# Gollek Optimization Plugin Core

Core SPI and plugin manager for GPU kernel optimization plugins.

## Overview

This module provides the core infrastructure for the Gollek optimization plugin system, including:

- **OptimizationPlugin SPI**: Interface for implementing optimization plugins
- **OptimizationPluginManager**: Singleton manager for plugin lifecycle
- **ExecutionContext**: Context provided to plugins during execution

## Installation

```bash
cd inference-gollek/core/gollek-plugin-optimization-core
mvn clean install
```

## Usage

### Registering Plugins

```java
import tech.kayys.gollek.plugin.optimization.OptimizationPluginManager;
import tech.kayys.gollek.plugin.optimization.OptimizationPlugin;

// Get manager instance
OptimizationPluginManager manager = OptimizationPluginManager.getInstance();

// Register plugins
manager.register(new FlashAttention3Plugin());
manager.register(new PagedAttentionPlugin());

// Initialize with configuration
Map<String, Object> config = loadConfig();
manager.initialize(config);

// Apply optimizations during inference
ExecutionContext context = createExecutionContext();
List<String> applied = manager.applyOptimizations(context);
```

### Implementing a Plugin

```java
import tech.kayys.gollek.plugin.optimization.*;
import java.util.*;

public class MyOptimizationPlugin implements OptimizationPlugin {
    
    @Override
    public String id() {
        return "my-optimization";
    }
    
    @Override
    public String name() {
        return "My Optimization";
    }
    
    @Override
    public String description() {
        return "Custom GPU optimization";
    }
    
    @Override
    public boolean isAvailable() {
        // Check hardware requirements
        return GpuDetector.isNvidia() && 
               GpuDetector.getComputeCapability() >= 80;
    }
    
    @Override
    public int priority() {
        return 50; // Higher values execute first
    }
    
    @Override
    public boolean apply(ExecutionContext context) {
        // Get model parameters
        int hiddenSize = context.getParameter("hidden_size", 4096);
        
        // Get memory buffers
        MemoryBuffer qBuffer = context.getBuffer("q").orElse(null);
        
        // Apply optimization
        return applyKernel(qBuffer.getPointer(), ...);
    }
}
```

## Architecture

```
┌─────────────────────────────────────────┐
│      Inference Engine                   │
└─────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────┐
│  OptimizationPluginManager              │
│  - Plugin discovery                     │
│  - Lifecycle management                 │
│  - Hardware detection                   │
│  - Priority-based execution             │
└─────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────┐
│      Optimization Plugins               │
│  - FlashAttention-3                     │
│  - FlashAttention-4                     │
│  - PagedAttention                       │
│  - KV Cache                             │
│  - ...                                  │
└─────────────────────────────────────────┘
```

## Components

### OptimizationPlugin

Main interface for optimization plugins:

- `id()` - Unique plugin identifier
- `name()` - Human-readable name
- `description()` - Plugin description
- `isAvailable()` - Hardware compatibility check
- `priority()` - Execution priority (higher = first)
- `apply(context)` - Apply optimization
- `shutdown()` - Cleanup resources

### OptimizationPluginManager

Singleton manager handling:

- Plugin registration and unregistration
- Hardware capability detection
- Priority-based execution ordering
- Health monitoring
- Configuration management

### ExecutionContext

Provides plugins with:

- Model parameters (hidden size, num heads, etc.)
- Memory buffers (Q, K, V matrices)
- GPU stream handles (CUDA/HIP)
- Execution phase information
- Context attributes

## Configuration

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
  }
}
```

## Dependencies

- JNA 5.14.0+ for native library binding
- JBoss Logging for logging

## Building

```bash
# Build core module
mvn clean install

# Build with tests
mvn clean test

# Build JAR
mvn clean package
```

## Integration

### With Provider Plugins

```java
// In your provider implementation
import tech.kayys.gollek.plugin.optimization.OptimizationPluginManager;

public class MyProvider implements LLMProvider {
    
    private final OptimizationPluginManager optimizationManager = 
        OptimizationPluginManager.getInstance();
    
    @Override
    public InferenceResponse infer(ProviderRequest request) {
        // Create execution context
        ExecutionContext context = createExecutionContext(request);
        
        // Apply optimizations
        List<String> applied = optimizationManager.applyOptimizations(context);
        
        // Execute inference with optimizations applied
        return executeInference(context);
    }
}
```

### With GPU Kernels

```java
// In GPU kernel runner
import tech.kayys.gollek.plugin.optimization.OptimizationPlugin;

public class CudaRunner {
    
    public void execute(ExecutionContext context) {
        // Get available optimizations for current GPU
        List<OptimizationPlugin> available = 
            OptimizationPluginManager.getInstance().getAvailablePlugins();
        
        // Apply each optimization
        for (OptimizationPlugin plugin : available) {
            if (plugin.apply(context)) {
                LOG.infof("Applied optimization: %s", plugin.id());
            }
        }
        
        // Execute kernel with optimizations applied
        executeKernel(context);
    }
}
```

## Plugin Discovery

Plugins are auto-discovered from:

1. Classpath (JAR dependencies)
2. `~/.gollek/plugins/optimization/` directory
3. Explicit registration via `manager.register(plugin)`

## Lifecycle

1. **Discovery**: Plugins discovered from classpath/directory
2. **Registration**: Plugins registered with manager
3. **Initialization**: `initialize(config)` called at startup
4. **Availability Check**: `isAvailable()` checks hardware
5. **Application**: `apply(context)` during inference
6. **Shutdown**: `shutdown()` on engine shutdown

## Best Practices

### Thread Safety

Plugin methods should be thread-safe:

```java
@Override
public boolean apply(ExecutionContext context) {
    synchronized (this) {
        // Thread-safe kernel application
        return applyKernel(...);
    }
}
```

### Error Handling

Gracefully handle errors:

```java
@Override
public boolean apply(ExecutionContext context) {
    try {
        return applyKernel(...);
    } catch (Exception e) {
        LOG.errorf("Optimization failed: %s", e.getMessage());
        return false; // Fallback to standard kernel
    }
}
```

### Resource Management

Clean up resources on shutdown:

```java
@Override
public void shutdown() {
    if (nativeBuffer != null) {
        nativeBuffer.close();
    }
    if (cudaStream != 0) {
        CudaBindings.cudaStreamDestroy(cudaStream);
    }
}
```

## Testing

```java
@Test
void testOptimizationPlugin() {
    OptimizationPlugin plugin = new FlashAttention3Plugin();
    
    // Test availability
    boolean available = plugin.isAvailable();
    
    // Test application
    if (available) {
        ExecutionContext context = createMockContext();
        boolean applied = plugin.apply(context);
        assertTrue(applied);
    }
}
```

## Resources

- [Optimization Plugins Documentation](https://gollek-ai.github.io/docs/optimization-plugins)
- [GPU Kernels Documentation](https://gollek-ai.github.io/docs/gpu-kernels)
- [FlashAttention-3 Paper](https://arxiv.org/abs/2307.08691)
- [PagedAttention Paper](https://arxiv.org/abs/2309.06180)

## License

MIT License - See LICENSE file for details.
