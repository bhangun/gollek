# Runner Plugin Integration Guide

## Overview

This guide describes how to integrate the existing GGUF runner implementation (`gollek-ext-runner-gguf`) with the new runner plugin system and update the engine to use it.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              GollekEngine                               │
│  (DefaultInferenceEngine)                               │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│           InferenceService                              │
│  - inferAsync()                                         │
│  - streamAsync()                                        │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│        GollekProviderRegistry                           │
│  - discoverProviders()                                  │
│  - register()                                           │
│  - getProvider()                                        │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│         RunnerPluginRegistry (NEW)                      │
│  - discoverRunners()                                    │
│  - registerRunner()                                     │
│  - createSession()                                      │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│           Runner Plugins                                │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                │
│  │  GGUF    │ │   ONNX   │ │TensorRT  │                │
│  │ Runner   │ │  Runner  │ │  Runner  │                │
│  └──────────┘ └──────────┘ └──────────┘                │
│       ↓              ↓             ↓                    │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                │
│  │LlamaCpp  │ │ Onnx     │ │ TensorRT │                │
│  │ Runner   │ │ Runtime  │ │  Engine  │                │
│  │(existing)│ │(existing)│ │(existing)│                │
│  └──────────┘ └──────────┘ └──────────┘                │
└─────────────────────────────────────────────────────────┘
```

## Integration Steps

### Step 1: Create RunnerPluginRegistry

```java
package tech.kayys.gollek.engine.registry;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Instance;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.gollek.plugin.runner.RunnerPlugin;
import tech.kayys.gollek.plugin.runner.RunnerSession;
import tech.kayys.gollek.plugin.runner.RunnerPluginManager;

import java.util.Map;
import java.util.Optional;

@ApplicationScoped
public class RunnerPluginRegistry {

    private static final Logger LOG = Logger.getLogger(RunnerPluginRegistry.class);

    @Inject
    Instance<RunnerPlugin> runnerPluginInstances;

    private final RunnerPluginManager pluginManager = RunnerPluginManager.getInstance();

    @jakarta.annotation.PostConstruct
    public void discoverRunners() {
        LOG.info("Discovering runner plugins...");

        runnerPluginInstances.stream()
                .filter(RunnerPlugin::isAvailable)
                .forEach(plugin -> {
                    try {
                        pluginManager.register(plugin);
                        LOG.infof("Registered runner plugin: %s", plugin.id());
                    } catch (Exception e) {
                        LOG.errorf(e, "Failed to register runner plugin: %s", plugin.id());
                    }
                });

        LOG.infof("Runner plugin discovery complete. Total: %d", 
                  pluginManager.getAllPlugins().size());
    }

    public Optional<RunnerSession> createSession(String modelPath, Map<String, Object> config) {
        return pluginManager.createSession(modelPath, config);
    }

    public void initialize(Map<String, Object> config) {
        pluginManager.initialize(config);
    }

    public void shutdown() {
        pluginManager.shutdown();
    }
}
```

### Step 2: Update GGUFProvider to Use Runner Plugin

```java
@ApplicationScoped
public class GGUFProvider implements StreamingProvider {

    @Inject
    RunnerPluginRegistry runnerRegistry;

    private RunnerSession createSessionForModel(String modelPath) {
        return runnerRegistry.createSession(modelPath, config.toMap())
                .orElseThrow(() -> new ProviderException.ProviderInitializationException(
                    PROVIDER_ID, "No runner available for model: " + modelPath));
    }

    @Override
    public Uni<InferenceResponse> infer(ProviderRequest request) {
        // Get or create session
        RunnerSession session = getOrCreateSession(request.getModel());
        
        // Convert ProviderRequest to InferenceRequest
        InferenceRequest inferenceRequest = convert(request);
        
        // Execute via runner session
        return session.infer(inferenceRequest);
    }

    @Override
    public Multi<InferenceChunk> inferStream(ProviderRequest request) {
        RunnerSession session = getOrCreateSession(request.getModel());
        InferenceRequest inferenceRequest = convert(request);
        return session.stream(inferenceRequest);
    }
}
```

### Step 3: Update GGUFRunnerPlugin to Wrap LlamaCppRunner

```java
public class GGUFRunnerPlugin implements RunnerPlugin {

    private final GGUFProviderConfig config;
    private final LlamaCppBinding binding;

    @Inject
    public GGUFRunnerPlugin(GGUFProviderConfig config, LlamaCppBinding binding) {
        this.config = config;
        this.binding = binding;
    }

    @Override
    public RunnerSession createSession(String modelPath, Map<String, Object> config) {
        // Create LlamaCppRunner instance
        LlamaCppRunner runner = new LlamaCppRunner(
            this.config, 
            binding,
            sessionManager,
            meterRegistry,
            tracer
        );
        
        // Wrap in session
        return new LlamaCppRunnerSession(runner, modelPath, config);
    }
}
```

### Step 4: Update Engine Bootstrap

```java
@ApplicationScoped
public class InferenceEngineBootstrap {

    @Inject
    RunnerPluginRegistry runnerRegistry;

    @Inject
    GollekProviderRegistry providerRegistry;

    public void onStart(@Observes StartupEvent event) {
        // Initialize runner plugins first
        runnerRegistry.discoverRunners();
        runnerRegistry.initialize(loadConfig());

        // Then initialize providers (which may use runners)
        providerRegistry.discoverProviders();
    }

    public @jakarta.annotation.PreDestroy
    public void shutdown() { {
        runnerRegistry.shutdown();
    }
}
```

### Step 5: Update POM Dependencies

```xml
<!-- In gollek-engine/pom.xml -->
<dependency>
    <groupId>tech.kayys.gollek</groupId>
    <artifactId>gollek-plugin-runner-core</artifactId>
    <version>${project.version}</version>
</dependency>

<dependency>
    <groupId>tech.kayys.gollek</groupId>
    <artifactId>gollek-plugin-runner-gguf</artifactId>
    <version>${project.version}</version>
</dependency>

<dependency>
    <groupId>tech.kayys.gollek</groupId>
    <artifactId>gollek-ext-runner-gguf</artifactId>
    <version>${project.version}</version>
</dependency>
```

## Benefits

### Before (Direct Coupling)
```java
// Engine directly instantiates runners
LlamaCppRunner runner = new LlamaCppRunner(config, binding, ...);
```

**Issues:**
- Tight coupling to specific runner implementation
- Hard to add new runners
- All runners loaded even if not used
- No hot-reload capability

### After (Plugin-Based)
```java
// Engine uses runner plugin registry
Optional<RunnerSession> session = runnerRegistry.createSession(modelPath, config);
```

**Benefits:**
- Loose coupling via plugin SPI
- Easy to add new runners
- Selective loading (only needed runners)
- Hot-reload support
- Better resource management

## Migration Path

### Phase 1: Parallel Operation
- Keep existing GGUFProvider working
- Add RunnerPluginRegistry alongside
- Test both paths

### Phase 2: Gradual Migration
- Migrate GGUFProvider to use runner plugins
- Keep old path as fallback
- Monitor performance

### Phase 3: Full Integration
- Remove old direct instantiation
- All runners use plugin system
- Enable hot-reload features

## Configuration

```yaml
gollek:
  runners:
    gguf-runner:
      enabled: true
      n_gpu_layers: -1
      n_ctx: 4096
      flash_attn: true
    onnx-runner:
      enabled: false  # Not loaded
    tensorrt-runner:
      enabled: false  # Not loaded
```

## Testing

```java
@QuarkusTest
public class RunnerPluginIntegrationTest {

    @Inject
    RunnerPluginRegistry runnerRegistry;

    @Test
    public void testGGUFRunnerPlugin() {
        Optional<RunnerSession> session = runnerRegistry.createSession(
            "models/llama-3-8b.gguf", 
            Map.of()
        );

        assertTrue(session.isPresent());
        
        InferenceResponse response = session.get()
            .infer(request)
            .await()
            .atMost(Duration.ofSeconds(30));

        assertNotNull(response.getContent());
    }
}
```

## Resources

- Runner Plugin Core: `inference-gollek/core/gollek-plugin-runner-core/`
- GGUF Runner Plugin: `inference-gollek/plugins/runner/gguf/gollek-plugin-runner-gguf/`
- Existing GGUF Implementation: `inference-gollek/plugins/runner/gguf/gollek-ext-runner-gguf/`
- Engine: `inference-gollek/core/gollek-engine/`
