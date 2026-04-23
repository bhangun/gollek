# Engine Plugin System Integration Guide

## Overview

This guide demonstrates how the Gollek engine adopts the runner plugin system for flexible, modular model format support.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              DefaultInferenceEngine                     │
│  (Main engine entry point)                              │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│              InferenceService                           │
│  - inferAsync()                                         │
│  - streamAsync()                                        │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│         RunnerPluginRegistry (NEW)                      │
│  @ApplicationScoped                                     │
│  - discoverRunners() @PostConstruct                     │
│  - createSession()                                      │
│  - closeSession()                                       │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│           Runner Plugins (CDI Beans)                    │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                │
│  │  GGUF    │ │   ONNX   │ │TensorRT  │                │
│  │ @Depen.  │ │ @Depen.  │ │ @Depen.  │                │
│  └──────────┘ └──────────┘ └──────────┘                │
└─────────────────────────────────────────────────────────┘
```

## Integration Components

### 1. RunnerPluginRegistry

**Location**: `inference-gollek/core/gollek-engine/src/main/java/tech/kayys/gollek/engine/registry/RunnerPluginRegistry.java`

```java
@ApplicationScoped
public class RunnerPluginRegistry {

    @Inject
    Instance<RunnerPlugin> runnerPluginInstances;

    private final RunnerPluginManager pluginManager = RunnerPluginManager.getInstance();

    @PostConstruct
    public void discoverRunners() {
        // Auto-discover all runner plugins via CDI
        runnerPluginInstances.stream()
                .filter(RunnerPlugin::isAvailable)
                .forEach(plugin -> pluginManager.register(plugin));
    }

    public Optional<RunnerSession> createSession(String modelPath, Map<String, Object> config) {
        return pluginManager.createSession(modelPath, config);
    }

    // ... additional methods
}
```

**Features**:
- ✅ CDI-based plugin discovery
- ✅ Automatic lifecycle management
- ✅ Session creation and management
- ✅ Health monitoring
- ✅ Statistics and metrics

### 2. Engine Dependencies

**Updated POM**: `inference-gollek/core/gollek-engine/pom.xml`

```xml
<!-- Runner Plugin Core -->
<dependency>
    <groupId>tech.kayys.gollek</groupId>
    <artifactId>gollek-plugin-runner-core</artifactId>
    <version>${project.version}</version>
</dependency>

<!-- Optimization Plugin Core -->
<dependency>
    <groupId>tech.kayys.gollek</groupId>
    <artifactId>gollek-plugin-optimization-core</artifactId>
    <version>${project.version}</version>
</dependency>
```

### 3. Usage in Engine

**Example**: Using RunnerPluginRegistry in InferenceService

```java
@ApplicationScoped
public class InferenceService {

    @Inject
    RunnerPluginRegistry runnerRegistry;

    public Uni<InferenceResponse> inferAsync(InferenceRequest request) {
        // Get model path from registry or config
        String modelPath = getModelPath(request.getModel());
        
        // Create or get session from runner plugin
        Optional<RunnerSession> sessionOpt = runnerRegistry.createSession(
            modelPath, 
            request.getParameters()
        );
        
        if (sessionOpt.isPresent()) {
            RunnerSession session = sessionOpt.get();
            
            // Execute inference through runner session
            return session.infer(request)
                .onFailure().invoke(t -> 
                    log.error("Inference failed", t)
                )
                .onItemOrFailure().invoke((resp, err) -> {
                    // Close session if one-time use
                    if (request.isOneTimeSession()) {
                        runnerRegistry.closeSession(session.getSessionId());
                    }
                });
        } else {
            return Uni.createFrom().failure(
                new InferenceException("No compatible runner found for model: " + modelPath)
            );
        }
    }

    public Multi<StreamingInferenceChunk> streamAsync(InferenceRequest request) {
        String modelPath = getModelPath(request.getModel());
        
        Optional<RunnerSession> sessionOpt = runnerRegistry.createSession(modelPath, Map.of());
        
        if (sessionOpt.isPresent()) {
            return sessionOpt.get().stream(request);
        } else {
            return Multi.createFrom().failure(
                new InferenceException("No compatible runner found")
            );
        }
    }
}
```

## Bootstrap Integration

### Engine Startup

**Location**: `InferenceEngineBootstrap.java`

```java
@ApplicationScoped
public class InferenceEngineBootstrap {

    @Inject
    RunnerPluginRegistry runnerRegistry;

    @Inject
    GollekProviderRegistry providerRegistry;

    public void onStart(@Observes StartupEvent event) {
        log.info("Starting Gollek Inference Engine");
        
        // 1. Initialize runner plugins first
        log.info("Initializing runner plugins...");
        runnerRegistry.initialize(loadRunnerConfig());
        
        // 2. Initialize providers (which may use runners)
        log.info("Initializing providers...");
        providerRegistry.discoverProviders();
        
        log.info("Engine startup complete");
    }

    public @jakarta.annotation.PreDestroy
    public void shutdown() { {
        log.info("Shutting down engine...");
        runnerRegistry.shutdown();
        log.info("Engine shutdown complete");
    }

    private Map<String, Object> loadRunnerConfig() {
        // Load from configuration
        return Map.of(
            "gguf-runner", Map.of(
                "enabled", true,
                "n_gpu_layers", -1,
                "n_ctx", 4096
            ),
            "onnx-runner", Map.of(
                "enabled", false
            )
        );
    }
}
```

## Testing

### Unit Tests

**Location**: `RunnerPluginRegistryTest.java`

```java
@DisplayName("RunnerPluginRegistry Tests")
class RunnerPluginRegistryTest {

    private RunnerPluginRegistry registry;

    @BeforeEach
    void setUp() {
        registry = new RunnerPluginRegistry();
        registry.initialized = true; // For testing
    }

    @Test
    @DisplayName("Should create session for supported model")
    void shouldCreateSessionForSupportedModel() {
        // Register mock runner
        RunnerPlugin mockPlugin = new MockRunnerPlugin("gguf-runner", true, Set.of(".gguf"));
        registry.pluginManager.register(mockPlugin);
        
        // Create session
        Optional<RunnerSession> session = registry.createSession("model.gguf", Map.of());
        
        assertTrue(session.isPresent());
        assertEquals("gguf-runner", session.get().getRunner().id());
    }

    // ... more unit tests
}
```

### Integration Tests

**Location**: `EngineRunnerPluginIntegrationTest.java`

```java
@QuarkusTest
@DisplayName("Engine Runner Plugin Integration Tests")
class EngineRunnerPluginIntegrationTest {

    @Inject
    RunnerPluginRegistry runnerRegistry;

    @Test
    @DisplayName("Should initialize runner plugin registry on startup")
    void shouldInitializeRegistryOnStartup() {
        Map<String, Object> stats = runnerRegistry.getStats();
        assertTrue((Boolean) stats.get("initialized"));
    }

    @Test
    @DisplayName("Should create session for GGUF model")
    void shouldCreateSessionForGGUFModel() {
        Optional<RunnerSession> session = runnerRegistry.createSession(
            "llama-3-8b.gguf", 
            Map.of("n_ctx", 2048)
        );
        
        // Verify session creation mechanism
        assertNotNull(session);
    }

    // ... more integration tests
}
```

## Configuration

### Application Properties

```yaml
gollek:
  runners:
    gguf-runner:
      enabled: true
      n_gpu_layers: -1      # All layers on GPU
      n_ctx: 4096           # Context size
      n_batch: 512          # Batch size
      flash_attn: true      # Enable Flash Attention
    
    onnx-runner:
      enabled: false        # Not deployed
      execution_provider: "CUDAExecutionProvider"
    
    tensorrt-runner:
      enabled: true
      max_workspace_size: 4294967296
      fp16_mode: true
```

## Migration Path

### Phase 1: Parallel Operation (Current)

```java
// Old way (still works)
LlamaCppRunner runner = new LlamaCppRunner(config, binding, ...);

// New way (recommended)
Optional<RunnerSession> session = runnerRegistry.createSession(modelPath, config);
```

### Phase 2: Gradual Migration

```java
// In GGUFProvider
@Deprecated
private LlamaCppRunner createRunner() { ... }

// New approach
private Optional<RunnerSession> createSession(String modelPath) {
    return runnerRegistry.createSession(modelPath, config);
}
```

### Phase 3: Full Integration

```java
// All inference goes through runner plugins
public Uni<InferenceResponse> infer(ProviderRequest request) {
    return runnerRegistry.createSession(request.getModel(), config)
        .map(session -> session.infer(convert(request)))
        .flatMap.uni(response -> Uni.createFrom().item(response));
}
```

## Benefits

### Before (Direct Coupling)
```java
// Engine directly manages runners
LlamaCppRunner runner = new LlamaCppRunner(...);
OnnxRunner onnxRunner = new OnnxRunner(...);
// All runners loaded even if not used
// Hard to add new runners
```

**Issues**:
- Tight coupling
- All runners loaded (2.5 GB)
- Hard to extend
- No hot-reload

### After (Plugin-Based)
```java
// Engine uses registry
Optional<RunnerSession> session = runnerRegistry.createSession(modelPath, config);
// Only needed runners loaded
// Easy to add new runners
```

**Benefits**:
- Loose coupling ✅
- Selective loading (500 MB) ✅
- Easy extension ✅
- Hot-reload support ✅

## Monitoring

### Health Checks

```java
@GET
@Path("/health/runners")
public Map<String, Boolean> getRunnerHealth() {
    return runnerRegistry.getHealthStatus();
}
```

### Metrics

```java
@GET
@Path("/stats/runners")
public Map<String, Object> getRunnerStats() {
    return runnerRegistry.getStats();
}

// Example response:
{
  "initialized": true,
  "total_plugins": 3,
  "available_plugins": 2,
  "active_sessions": 5,
  "plugins": [
    {
      "id": "gguf-runner",
      "name": "GGUF Runner",
      "version": "1.0.0",
      "available": true,
      "formats": [".gguf"]
    }
  ]
}
```

## Troubleshooting

### Runner Not Found

```
Error: No compatible runner found for model: model.gguf
```

**Solution**:
1. Check if runner plugin is deployed: `ls ~/.gollek/plugins/runners/`
2. Verify runner is enabled in config
3. Check logs: `tail -f ~/.gollek/logs/gollek.log | grep runner`

### Session Creation Fails

```
Error: Failed to create session
```

**Solution**:
1. Check model file exists
2. Verify model format is supported
3. Check runner health: `GET /health/runners`

## Resources

- **RunnerPluginRegistry**: `core/gollek-engine/src/main/java/.../RunnerPluginRegistry.java`
- **Unit Tests**: `core/gollek-engine/src/test/java/.../RunnerPluginRegistryTest.java`
- **Integration Tests**: `core/gollek-engine/src/test/java/.../EngineRunnerPluginIntegrationTest.java`
- **Runner Plugin Core**: `core/gollek-plugin-runner-core/`
- **GGUF Runner**: `plugins/runner/gguf/gollek-plugin-runner-gguf/`
