



## Architecture Overview

### Design Principles

**1. Hexagonal Architecture (Ports & Adapters)**
- Core domain logic isolated from infrastructure concerns
- Adapters for different model formats (GGUF, ONNX, Triton, LLM Cloud Provider (OpenAI, Anthropic, Google), etc.)
- Easy to test and swap implementations

**2. Multi-Tenancy First**
- Tenant isolation at data, compute, and configuration levels
- Resource quotas and fair scheduling per tenant
- Secure tenant context propagation

**3. Build-Time Optimization**
- Maven profiles for CPU/GPU/TPU variants
- Quarkus build-time pruning with `@IfBuildProperty`
- GraalVM native image ready with reachability metadata

**4. Cloud-Native & Kubernetes-Ready**
- Health/readiness probes
- Graceful shutdown and resource cleanup
- Service mesh compatible (mTLS, traffic management)

---

## Project Structure

```
inference-platform/
├── pom.xml                                    # Parent POM with profiles
├── README.md
├── ARCHITECTURE.md
├── docker/
│   ├── Dockerfile.jvm-cpu
│   ├── Dockerfile.jvm-gpu
│   ├── Dockerfile.native-cpu
│   └── Dockerfile.native-gpu
├── k8s/
│   ├── base/                                  # Kustomize base
│   ├── overlays/
│   │   ├── dev/
│   │   ├── staging/
│   │   └── production/
│   └── helm/                                  # Helm charts
├── inference-api/                             # API contracts & DTOs
│   ├── pom.xml
│   └── src/main/java/
│       └── com/enterprise/inference/api/
│           ├── InferenceRequest.java
│           ├── InferenceResponse.java
│           ├── ModelMetadata.java
│           └── TenantContext.java
├── inference-core/                            # Domain logic & SPI
│   ├── pom.xml
│   └── src/main/java/
│       └── com/enterprise/inference/core/
│           ├── domain/
│           │   ├── ModelManifest.java
│           │   ├── ModelVersion.java
│           │   ├── ResourceRequirements.java
│           │   └── InferenceSession.java
│           ├── ports/
│           │   ├── inbound/
│           │   │   ├── InferenceUseCase.java
│           │   │   └── ModelManagementUseCase.java
│           │   └── outbound/
│           │       ├── ModelRunner.java        # Core SPI
│           │       ├── ModelRepository.java
│           │       ├── MetricsPublisher.java
│           │       └── TenantResolver.java
│           ├── service/
│           │   ├── InferenceOrchestrator.java
│           │   ├── ModelRouterService.java
│           │   ├── SelectionPolicy.java
│           │   └── FallbackStrategy.java
│           └── exceptions/
│               ├── ModelNotFoundException.java
│               ├── InferenceException.java
│               └── TenantQuotaExceededException.java
├── inference-adapter-gguf/                    # llama.cpp adapter
│   ├── pom.xml
│   └── src/main/java/
│       └── com/enterprise/inference/adapter/gguf/
│           ├── LlamaCppRunner.java
│           ├── GGUFModelLoader.java
│           ├── FFMNativeBinding.java
│           └── resources/
│               └── META-INF/
│                   └── native-image/
│                       └── reflect-config.json
├── inference-adapter-onnx/                    # ONNX Runtime adapter
│   ├── pom.xml
│   └── src/main/java/
│       └── com/enterprise/inference/adapter/onnx/
│           ├── OnnxRuntimeRunner.java
│           ├── OnnxSessionManager.java
│           ├── ExecutionProviderSelector.java
│           └── TensorConverter.java
├── inference-adapter-triton/                  # Triton Inference Server adapter
│   ├── pom.xml
│   └── src/main/java/
│       └── com/enterprise/inference/adapter/triton/
│           ├── TritonGrpcRunner.java
│           ├── TritonHttpRunner.java
│           ├── TritonClientPool.java
│           └── proto/                         # Generated gRPC stubs
├── inference-adapter-tpu/                     # Google Cloud TPU adapter
│   ├── pom.xml
│   └── src/main/java/
│       └── com/enterprise/inference/adapter/tpu/
│           └── TpuRunner.java
├── inference-infrastructure/                  # Infrastructure adapters
│   ├── pom.xml
│   └── src/main/java/
│       └── com/enterprise/inference/infrastructure/
│           ├── persistence/
│           │   ├── S3ModelRepository.java
│           │   ├── MinIOModelRepository.java
│           │   └── PostgresMetadataStore.java
│           ├── messaging/
│           │   ├── KafkaEventPublisher.java
│           │   └── events/
│           ├── security/
│           │   ├── JwtTenantResolver.java
│           │   ├── KeycloakIntegration.java
│           │   └── TenantContextInterceptor.java
│           └── observability/
│               ├── PrometheusMetrics.java
│               ├── OpenTelemetryTracing.java
│               └── StructuredLogging.java
├── inference-runtime/                     # Quarkus application assembly
│   ├── pom.xml
│   └── src/main/
│       ├── java/
│       │   └── com/enterprise/inference/runtime/
│       │       ├── rest/
│       │       │   ├── InferenceResource.java
│       │       │   ├── ModelManagementResource.java
│       │       │   ├── HealthResource.java
│       │       │   └── filters/
│       │       │       ├── TenantFilter.java
│       │       │       └── RateLimitFilter.java
│       │       ├── config/
│       │       │   ├── InferenceConfig.java
│       │       │   ├── TenantConfig.java
│       │       │   └── RunnerConfig.java
│       │       └── lifecycle/
│       │           ├── StartupInitializer.java
│       │           └── ShutdownHandler.java
│       └── resources/
│           ├── application.yml
│           ├── application-dev.yml
│           ├── application-prod.yml
│           └── META-INF/
│               └── microprofile-config.properties
└── inference-tests/
    ├── pom.xml
    └── src/test/
        ├── java/
        │   ├── integration/
        │   ├── performance/
        │   └── e2e/
        └── resources/
            └── fixtures/
```

---

## Core Components

### 1. Core Domain Model

```java
// inference-core/src/main/java/com/enterprise/inference/core/domain/

/**
 * Immutable model manifest representing all metadata and artifacts
 * for a specific model version.
 */
public record ModelManifest(
    String modelId,
    String name,
    String version,
    TenantId tenantId,
    Map<ModelFormat, ArtifactLocation> artifacts,
    List<SupportedDevice> supportedDevices,
    ResourceRequirements resourceRequirements,
    Map<String, Object> metadata,
    Instant createdAt,
    Instant updatedAt
) {
    public boolean supportsFormat(ModelFormat format) {
        return artifacts.containsKey(format);
    }
    
    public boolean supportsDevice(DeviceType deviceType) {
        return supportedDevices.stream()
            .anyMatch(d -> d.type() == deviceType);
    }
}

/**
 * Value object for tenant identification and isolation
 */
public record TenantId(String value) {
    public TenantId {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException("TenantId cannot be empty");
        }
    }
}

/**
 * Resource requirements and constraints for model execution
 */
public record ResourceRequirements(
    MemorySize minMemory,
    MemorySize recommendedMemory,
    MemorySize minVRAM,
    Optional<Integer> minCores,
    Optional<DiskSpace> diskSpace
) {}

/**
 * Enumeration of supported model formats
 */
public enum ModelFormat {
    GGUF("gguf", "llama.cpp"),
    ONNX("onnx", "ONNX Runtime"),
    TENSORRT("trt", "TensorRT"),
    TORCHSCRIPT("pt", "TorchScript"),
    TENSORFLOW_SAVED_MODEL("pb", "TensorFlow");
    
    private final String extension;
    private final String runtime;
    
    ModelFormat(String extension, String runtime) {
        this.extension = extension;
        this.runtime = runtime;
    }
}

/**
 * Device type enumeration with capability flags
 */
public enum DeviceType {
    CPU(false, false),
    CUDA(true, false),
    ROCM(true, false),
    METAL(true, false),
    TPU(false, true),
    OPENVINO(true, false);
    
    private final boolean supportsGpu;
    private final boolean supportsTpu;
}
```

### 2. Core Port Definitions (SPI)

```java
// inference-core/src/main/java/com/enterprise/inference/core/ports/outbound/

/**
 * Core SPI for model execution backends.
 * All adapters must implement this interface.
 */
public interface ModelRunner extends AutoCloseable {
    
    /**
     * Initialize the runner with model manifest and configuration
     * @param manifest Model metadata and artifact locations
     * @param config Runner-specific configuration
     * @param tenantContext Current tenant context for isolation
     * @throws ModelLoadException if initialization fails
     */
    void initialize(
        ModelManifest manifest, 
        Map<String, Object> config,
        TenantContext tenantContext
    ) throws ModelLoadException;
    
    /**
     * Execute synchronous inference
     * @param request Inference request with inputs
     * @param context Request context with timeout, priority, etc.
     * @return Inference response with outputs
     * @throws InferenceException if execution fails
     */
    InferenceResponse infer(
        InferenceRequest request,
        RequestContext context
    ) throws InferenceException;
    
    /**
     * Execute asynchronous inference with callback
     * @param request Inference request
     * @param context Request context
     * @return CompletionStage for async processing
     */
    CompletionStage<InferenceResponse> inferAsync(
        InferenceRequest request,
        RequestContext context
    );
    
    /**
     * Health check for this runner instance
     * @return Health status with diagnostics
     */
    HealthStatus health();
    
    /**
     * Get current resource utilization metrics
     * @return Resource usage snapshot
     */
    ResourceMetrics getMetrics();
    
    /**
     * Warm up the model (optional optimization)
     * @param sampleInputs Sample inputs for warming
     */
    default void warmup(List<InferenceRequest> sampleInputs) {
        // Default no-op
    }
    
    /**
     * Get runner metadata
     * @return Metadata about this runner implementation
     */
    RunnerMetadata metadata();
    
    /**
     * Gracefully release resources
     */
    @Override
    void close();
}

/**
 * Runner metadata for selection and diagnostics
 */
public record RunnerMetadata(
    String name,
    String version,
    List<ModelFormat> supportedFormats,
    List<DeviceType> supportedDevices,
    ExecutionMode executionMode,
    Map<String, Object> capabilities
) {}

/**
 * Repository for model artifacts and metadata
 */
public interface ModelRepository {
    
    /**
     * Load model manifest by ID
     */
    Optional<ModelManifest> findById(String modelId, TenantId tenantId);
    
    /**
     * List all models for tenant
     */
    List<ModelManifest> findByTenant(TenantId tenantId, Pageable pageable);
    
    /**
     * Save or update model manifest
     */
    ModelManifest save(ModelManifest manifest);
    
    /**
     * Download model artifact to local cache
     */
    Path downloadArtifact(
        ModelManifest manifest, 
        ModelFormat format
    ) throws ArtifactDownloadException;
    
    /**
     * Check if artifact is cached locally
     */
    boolean isCached(String modelId, ModelFormat format);
    
    /**
     * Evict artifact from local cache
     */
    void evictCache(String modelId, ModelFormat format);
}
```

### 3. Model Router & Selection Policy

```java
// inference-core/src/main/java/com/enterprise/inference/core/service/

/**
 * Orchestrates inference requests across multiple runners with
 * intelligent routing, fallback, and load balancing.
 */
@ApplicationScoped
public class InferenceOrchestrator {
    
    private final ModelRouterService router;
    private final ModelRunnerFactory factory;
    private final ModelRepository repository;
    private final MetricsPublisher metrics;
    private final CircuitBreaker circuitBreaker;
    
    @Inject
    public InferenceOrchestrator(
        ModelRouterService router,
        ModelRunnerFactory factory,
        ModelRepository repository,
        MetricsPublisher metrics,
        CircuitBreaker circuitBreaker
    ) {
        this.router = router;
        this.factory = factory;
        this.repository = repository;
        this.metrics = metrics;
        this.circuitBreaker = circuitBreaker;
    }
    
    /**
     * Execute inference with automatic runner selection and fallback
     */
    public InferenceResponse execute(
        String modelId,
        InferenceRequest request,
        TenantContext tenantContext
    ) {
        var span = Span.current();
        span.setAttribute("model.id", modelId);
        span.setAttribute("tenant.id", tenantContext.tenantId().value());
        
        // Load model manifest
        ModelManifest manifest = repository
            .findById(modelId, tenantContext.tenantId())
            .orElseThrow(() -> new ModelNotFoundException(modelId));
        
        // Build request context with timeout and priority
        RequestContext ctx = RequestContext.builder()
            .tenantContext(tenantContext)
            .timeout(Duration.ofSeconds(30))
            .priority(request.priority())
            .preferredDevice(request.deviceHint())
            .build();
        
        // Select and rank candidate runners
        List<RunnerCandidate> candidates = router.selectRunners(
            manifest, 
            ctx
        );
        
        InferenceException lastError = null;
        
        // Attempt inference with fallback
        for (RunnerCandidate candidate : candidates) {
            try {
                return executeWithRunner(
                    manifest, 
                    candidate, 
                    request, 
                    ctx
                );
            } catch (InferenceException e) {
                lastError = e;
                metrics.recordFailure(
                    candidate.runnerName(), 
                    modelId, 
                    e.getClass().getSimpleName()
                );
                
                // Don't retry on quota or validation errors
                if (e instanceof TenantQuotaExceededException ||
                    e instanceof ValidationException) {
                    throw e;
                }
                
                span.addEvent("Runner failed, attempting fallback", 
                    Attributes.of(
                        AttributeKey.stringKey("runner"), candidate.runnerName(),
                        AttributeKey.stringKey("error"), e.getMessage()
                    ));
            }
        }
        
        throw new AllRunnersFailedException(
            "All runners failed for model " + modelId, 
            lastError
        );
    }
    
    private InferenceResponse executeWithRunner(
        ModelManifest manifest,
        RunnerCandidate candidate,
        InferenceRequest request,
        RequestContext ctx
    ) {
        var timer = metrics.startTimer();
        
        try {
            // Get or create runner instance
            ModelRunner runner = factory.getRunner(
                manifest, 
                candidate.runnerName(),
                ctx.tenantContext()
            );
            
            // Execute with circuit breaker protection
            InferenceResponse response = circuitBreaker.call(
                () -> runner.infer(request, ctx)
            );
            
            metrics.recordSuccess(
                candidate.runnerName(), 
                manifest.modelId(), 
                timer.stop()
            );
            
            return response;
            
        } catch (Exception e) {
            metrics.recordFailure(
                candidate.runnerName(), 
                manifest.modelId(), 
                e.getClass().getSimpleName()
            );
            throw new InferenceException(
                "Inference failed with runner: " + candidate.runnerName(), 
                e
            );
        }
    }
}

/**
 * Selection policy implementation with scoring algorithm
 */
@ApplicationScoped
public class SelectionPolicy {
    
    private final RuntimeMetricsCache metricsCache;
    private final HardwareDetector hardwareDetector;
    
    /**
     * Rank available runners based on multiple criteria
     */
    public List<RunnerCandidate> rankRunners(
        ModelManifest manifest,
        RequestContext context,
        List<String> configuredRunners
    ) {
        List<RunnerCandidate> candidates = new ArrayList<>();
        
        // Get current hardware availability
        HardwareCapabilities hw = hardwareDetector.detect();
        
        for (String runnerName : configuredRunners) {
            RunnerMetadata runnerMeta = getRunnerMetadata(runnerName);
            
            // Filter by format compatibility
            if (!hasCompatibleFormat(manifest, runnerMeta)) {
                continue;
            }
            
            // Filter by device availability
            if (!isDeviceAvailable(runnerMeta, hw, context)) {
                continue;
            }
            
            // Calculate score
            int score = calculateScore(
                manifest, 
                runnerMeta, 
                context, 
                hw
            );
            
            candidates.add(new RunnerCandidate(
                runnerName, 
                score, 
                runnerMeta
            ));
        }
        
        // Sort by score descending
        candidates.sort(Comparator.comparing(
            RunnerCandidate::score
        ).reversed());
        
        return candidates;
    }
    
    /**
     * Multi-factor scoring algorithm
     */
    private int calculateScore(
        ModelManifest manifest,
        RunnerMetadata runner,
        RequestContext context,
        HardwareCapabilities hw
    ) {
        int score = 0;
        
        // Device preference match (highest weight)
        if (context.preferredDevice().isPresent() &&
            runner.supportedDevices().contains(
                context.preferredDevice().get()
            )) {
            score += 50;
        }
        
        // Format native support
        if (runner.supportedFormats().contains(
            manifest.artifacts().keySet().iterator().next()
        )) {
            score += 30;
        }
        
        // Historical performance (P95 latency)
        Optional<Duration> p95 = metricsCache.getP95Latency(
            runner.name(), 
            manifest.modelId()
        );
        if (p95.isPresent() && 
            p95.get().compareTo(context.timeout()) < 0) {
            score += 25;
        }
        
        // Resource availability
        if (hasAvailableResources(manifest, runner, hw)) {
            score += 20;
        }
        
        // Health status
        if (metricsCache.isHealthy(runner.name())) {
            score += 15;
        }
        
        // Cost optimization (favor CPU over GPU if performance OK)
        if (context.costSensitive() && 
            runner.supportedDevices().contains(DeviceType.CPU)) {
            score += 10;
        }
        
        // Current load (avoid overloaded runners)
        double currentLoad = metricsCache.getCurrentLoad(runner.name());
        if (currentLoad < 0.7) {
            score += 10;
        } else if (currentLoad > 0.9) {
            score -= 20;
        }
        
        return score;
    }
}
```

### 4. Runner Factory with Warm Pool

```java
// inference-core/src/main/java/com/enterprise/inference/core/service/

/**
 * Factory for creating and managing runner instances with
 * warm pool, lifecycle management, and tenant isolation.
 */
@ApplicationScoped
public class ModelRunnerFactory {
    
    private static final int MAX_POOL_SIZE = 10;
    private static final Duration IDLE_TIMEOUT = Duration.ofMinutes(15);
    
    @Inject
    Instance<ModelRunner> runnerProviders;
    
    @Inject
    ModelRepository repository;
    
    @Inject
    TenantConfig tenantConfig;
    
    // Pool: (tenantId, modelId, runnerName) -> Runner instance
    private final LoadingCache<RunnerCacheKey, ModelRunner> warmPool;
    
    // Track usage per runner for cleanup
    private final Map<RunnerCacheKey, Instant> lastAccess;
    
    @Inject
    public ModelRunnerFactory() {
        this.warmPool = Caffeine.newBuilder()
            .maximumSize(MAX_POOL_SIZE)
            .expireAfterAccess(IDLE_TIMEOUT)
            .removalListener(this::onRunnerEvicted)
            .build(this::createRunner);
            
        this.lastAccess = new ConcurrentHashMap<>();
        
        // Start cleanup scheduler
        startCleanupScheduler();
    }
    
    /**
     * Get or create runner instance for tenant
     */
    public ModelRunner getRunner(
        ModelManifest manifest,
        String runnerName,
        TenantContext tenantContext
    ) {
        RunnerCacheKey key = new RunnerCacheKey(
            tenantContext.tenantId(),
            manifest.modelId(),
            runnerName
        );
        
        // Update last access time
        lastAccess.put(key, Instant.now());
        
        // Get from pool or create
        return warmPool.get(key);
    }
    
    /**
     * Prewarm runners for specific models
     */
    public void prewarm(
        ModelManifest manifest,
        List<String> runnerNames,
        TenantContext tenantContext
    ) {
        runnerNames.forEach(runnerName -> {
            try {
                getRunner(manifest, runnerName, tenantContext);
            } catch (Exception e) {
                // Log but don't fail prewarming
                Log.warnf("Failed to prewarm runner %s: %s", 
                    runnerName, e.getMessage());
            }
        });
    }
    
    /**
     * Create new runner instance (called by cache loader)
     */
    private ModelRunner createRunner(RunnerCacheKey key) {
        // Load manifest
        ModelManifest manifest = repository
            .findById(key.modelId(), key.tenantId())
            .orElseThrow(() -> new ModelNotFoundException(key.modelId()));
        
        // Find runner provider by name
        ModelRunner runner = runnerProviders.stream()
            .filter(r -> r.metadata().name().equals(key.runnerName()))
            .findFirst()
            .orElseThrow(() -> new IllegalArgumentException(
                "Unknown runner: " + key.runnerName()
            ));
        
        // Get tenant-specific configuration
        Map<String, Object> config = tenantConfig.getRunnerConfig(
            key.tenantId(), 
            key.runnerName()
        );
        
        // Initialize runner
        TenantContext ctx = TenantContext.of(key.tenantId());
        runner.initialize(manifest, config, ctx);
        
        // Warmup if configured
        if (config.getOrDefault("warmup.enabled", false).equals(true)) {
            runner.warmup(Collections.emptyList());
        }
        
        Log.infof("Created runner %s for model %s (tenant %s)", 
            key.runnerName(), key.modelId(), key.tenantId().value());
        
        return runner;
    }
    
    /**
     * Cleanup callback when runner is evicted
     */
    private void onRunnerEvicted(
        RunnerCacheKey key, 
        ModelRunner runner,
        RemovalCause cause
    ) {
        if (runner != null) {
            try {
                runner.close();
                Log.infof("Closed runner %s for model %s (cause: %s)", 
                    key.runnerName(), key.modelId(), cause);
            } catch (Exception e) {
                Log.errorf(e, "Error closing runner %s", key.runnerName());
            }
        }
        lastAccess.remove(key);
    }
    
    /**
     * Background cleanup of idle runners
     */
    private void startCleanupScheduler() {
        Executors.newSingleThreadScheduledExecutor()
            .scheduleAtFixedRate(
                this::cleanupIdleRunners,
                5, 5, TimeUnit.MINUTES
            );
    }
    
    private void cleanupIdleRunners() {
        Instant threshold = Instant.now().minus(IDLE_TIMEOUT);
        
        lastAccess.entrySet().removeIf(entry -> {
            if (entry.getValue().isBefore(threshold)) {
                warmPool.invalidate(entry.getKey());
                return true;
            }
            return false;
        });
    }
    
    /**
     * Cache key for runner pooling
     */
    private record RunnerCacheKey(
        TenantId tenantId,
        String modelId,
        String runnerName
    ) {}
}
```

---

Below is a **focused, future-proof upgrade plan** centered on a **real plugin system**, long-term reliability, and avoiding architectural dead-ends.

I’ll structure this as:

1. 🔍 What’s already excellent (so we don’t break it)
2. 🧩 Missing plugin abstractions (core gap)
3. 🧠 Execution lifecycle plugins (reliability)
4. 🔐 Security & policy plugins (enterprise readiness)
5. 🧪 Validation & compatibility plugins (future-proofing)
6. ⚙️ Control-plane & hot-reload plugins
7. 📦 Packaging & versioning strategy
8. 🧭 Target end-state architecture (mental model)

---

## 1️⃣ What You Already Did Right (Don’t Touch These)

You already solved **hard problems** correctly:

✅ Hexagonal core with SPI (`ModelRunner`)
✅ Multi-format abstraction (GGUF / ONNX / Triton / TPU)
✅ Warm pool with eviction & lifecycle
✅ Intelligent routing with scoring & fallback
✅ Tenant-aware orchestration
✅ Metrics, circuit breakers, tracing hooks

⚠️ **Do NOT turn adapters into plugins directly** — that’s a trap.
Adapters are *implementations*, not *extension points*.

---

## 2️⃣ The Core Gap: No First-Class Plugin Contract

Right now:

* Plugins = CDI beans discovered at startup
* No lifecycle
* No compatibility checks
* No isolation
* No governance

### 🔧 Introduce a Real Plugin SPI (Non-negotiable)

Add a **core plugin abstraction**:

```java
public interface InferencePlugin {

    PluginDescriptor descriptor();

    /**
     * Called once at startup
     */
    void initialize(PluginContext context);

    /**
     * Called before each inference request
     */
    default void beforeInference(InferenceHookContext ctx) {}

    /**
     * Called after successful inference
     */
    default void afterInference(InferenceHookContext ctx, InferenceResponse response) {}

    /**
     * Called on inference failure
     */
    default void onFailure(InferenceHookContext ctx, Throwable error) {}

    /**
     * Health check for the plugin itself
     */
    default HealthStatus health() {
        return HealthStatus.healthy();
    }

    /**
     * Graceful shutdown
     */
    void shutdown();
}
```

```java
public record PluginDescriptor(
    String id,
    String name,
    String version,
    PluginType type,
    Set<PluginCapability> capabilities,
    SemanticVersion minEngineVersion,
    SemanticVersion maxEngineVersion
) {}
```

This makes plugins:

* Versioned
* Governed
* Observable
* Optional
* Replaceable

---

## 3️⃣ Execution Lifecycle Plugins (Reliability Boost)

Right now, orchestration logic is **hardcoded** in `InferenceOrchestrator`.

### Extract execution hooks

Introduce **execution phase plugins**:

```java
public enum InferencePhase {
    REQUEST_RECEIVED,
    MODEL_SELECTED,
    RUNNER_SELECTED,
    PRE_EXECUTION,
    POST_EXECUTION,
    RESPONSE_SERIALIZED
}
```

```java
public interface InferencePhasePlugin extends InferencePlugin {
    void onPhase(InferencePhase phase, InferenceHookContext ctx);
}
```

### What this enables

You can add plugins for:

* Retry policies
* Adaptive timeouts
* Shadow traffic
* Canary execution
* Request mutation
* Feature flags
* Chaos testing
* Rate limiting (remove from REST filter!)

💡 **Key idea**:

> The orchestrator should *emit events*, not *own behavior*.

---

## 4️⃣ Security & Policy as Plugins (Critical for Enterprise)

Right now:

* Security is infrastructure-bound
* Policies are implicit

### Add Policy Plugins

```java
public interface PolicyPlugin extends InferencePlugin {

    PolicyDecision evaluate(InferencePolicyContext ctx);
}
```

```java
public enum PolicyDecision {
    ALLOW,
    DENY,
    REQUIRE_APPROVAL,
    RATE_LIMIT
}
```

Use cases:

* Tenant quota enforcement
* Data residency rules
* Model usage permissions
* Cost ceilings
* Sensitive prompt blocking
* Regulated industry controls

⚠️ This keeps **security out of adapters and runners**.

---

## 5️⃣ Validation & Compatibility Plugins (Future-Proofing)

Today:

* Model compatibility logic is scattered
* No formal validation pipeline

### Add Model Validation Plugins

```java
public interface ModelValidationPlugin extends InferencePlugin {

    ValidationResult validate(ModelManifest manifest);
}
```

Examples:

* GGUF quant compatibility
* ONNX opset support
* GPU memory sufficiency
* Cross-version schema checks
* Deprecated format detection

This prevents:

* Bad model uploads
* Runtime crashes
* Silent performance degradation

---

## 6️⃣ Control Plane & Hot-Reconfiguration Plugins

Right now:

* Config changes require restart or redeploy
* No plugin-level config updates

### Introduce Dynamic Plugin Configuration

```java
public interface ConfigurablePlugin {

    void onConfigUpdate(Map<String, Object> newConfig);
}
```

Add a **PluginRegistry**:

```java
public interface PluginRegistry {

    List<InferencePlugin> all();

    <T extends InferencePlugin> List<T> byType(Class<T> type);

    Optional<InferencePlugin> byId(String id);

    void reload(String pluginId);
}
```

Now you can:

* Enable/disable plugins per tenant
* Roll out new policies live
* Inject A/B logic without downtime

---

## 7️⃣ Packaging & Versioning Strategy (This Is Huge)

### ❌ Avoid

* “Just another module”
* CDI-only discovery
* Fat JAR plugins

### ✅ Do This Instead

**Plugin packaging standard**:

```
inference-plugin-*.jar
└── META-INF/
    ├── inference-plugin.json
    └── services/
        └── com.enterprise.inference.plugin.InferencePlugin
```

```json
{
  "id": "cost-guard",
  "name": "Cost Guard Plugin",
  "version": "1.2.0",
  "type": "POLICY",
  "minEngineVersion": "2.0.0",
  "capabilities": ["RATE_LIMIT", "COST_CONTROL"]
}
```

Support:

* Semantic versioning
* Engine compatibility checks
* Controlled rollout
* Signed plugins (later)

---

## 8️⃣ Final Mental Model (Where This Ends Up)

Think of your system as:

```
┌────────────────────────────────────┐
│ Inference Engine Core              │
│                                    │
│  ┌──────────────┐                  │
│  │ Orchestrator │  ← emits phases  │
│  └──────┬───────┘                  │
│         │                          │
│  ┌──────▼────────┐                │
│  │ Plugin System │◄─────────────┐ │
│  └──────┬────────┘              │ │
│         │                        │ │
│ ┌───────▼───────┐   ┌──────────▼┐│
│ │ Policy Plugins│   │ Observers ││
│ └───────────────┘   └───────────┘│
│                                    │
│  ┌──────────────────────────────┐ │
│  │ ModelRunner Adapters (SPI)    │ │
│  │ GGUF | ONNX | Triton | Cloud  │ │
│  └──────────────────────────────┘ │
└────────────────────────────────────┘
```

Adapters **execute**
Plugins **govern, observe, and control**

---

## 🚀 Summary: What to Enhance

### High-impact improvements

✔ Add **InferencePlugin SPI**
✔ Move behavior into **execution phase plugins**
✔ Make **security & policy pluggable**
✔ Add **model validation plugins**
✔ Introduce **PluginRegistry + lifecycle**
✔ Support **dynamic config & hot reload**
✔ Formalize **plugin packaging & compatibility**



# 🧠 LLM Inference Kernel — FINAL RECAP

## 🎯 Purpose (What This Kernel Is)

A **request-scoped runtime** that:

* Accepts an inference request
* Runs it through **deterministic phases**
* Applies **plugins** (validation, policy, safety, transformation)
* Dispatches to an **LLM provider**
* Returns a **blocking or streaming response**

Nothing more.

---

## 🧱 Core Mental Model

```
InferenceEngine
  └── InferencePipeline
        └── InferencePhase[]
              └── InferencePhasePlugin[]
                    └── LLMProvider
```

Single request → linear execution → response.

---

## 1️⃣ InferenceEngine (Entry Point)

```java
public interface InferenceEngine {

    InferenceResponse infer(InferenceRequest request);
}
```

* Stateless
* Thread-safe
* One request in → one response out

---

## 2️⃣ InferenceRequest (Input)

```java
public final class InferenceRequest {

    private final String model;
    private final List<Message> messages;
    private final Map<String, Object> parameters;
    private final boolean streaming;
}
```

* Provider-agnostic
* Immutable
* Safe to log / audit

---

## 3️⃣ InferenceContext (Per Request)

```java
public interface InferenceContext {

    String requestId();

    InferenceRequest request();

    InferenceResponse response();

    Map<String, Object> attributes();

    void setResponse(InferenceResponse response);

    void fail(Throwable error);
}
```

* Exists **only during infer()**
* No persistence
* No resumption

---

## 4️⃣ InferencePhase (Deterministic Order)

```java
public enum InferencePhase {

    VALIDATION,
    PRE_PROCESSING,
    PROVIDER_DISPATCH,
    POST_PROCESSING;

    public static List<InferencePhase> ordered() {
        return List.of(values());
    }
}
```

* Linear
* No branching
* No looping

---

## 5️⃣ Plugin System (Strict & Minimal)

### Plugin Hierarchy (LOCKED)

```
Plugin
 └── ConfigurablePlugin
       └── InferencePhasePlugin
             └── ModelValidationPlugin
```

### Base Plugin

```java
public interface Plugin {

    String id();
    int order();

    default void initialize(EngineContext context) {}
    default void shutdown() {}
}
```

---

### InferencePhasePlugin

```java
public interface InferencePhasePlugin
        extends ConfigurablePlugin {

    InferencePhase phase();

    void execute(
        InferenceContext context,
        EngineContext engine
    );
}
```

* Phase-bound
* Deterministic
* No provider calls

---

## 6️⃣ InferencePipeline (Phase Executor)

```java
public interface InferencePipeline {

    void execute(InferenceContext context);
}
```

```java
public final class DefaultInferencePipeline
        implements InferencePipeline {

    private final Map<
        InferencePhase,
        List<InferencePhasePlugin>
    > plugins;

    @Override
    public void execute(InferenceContext context) {

        for (InferencePhase phase : InferencePhase.ordered()) {
            for (InferencePhasePlugin plugin : plugins.get(phase)) {
                plugin.execute(context, context.engine());
            }
        }
    }
}
```

---

## 7️⃣ LLM Provider Abstraction

### LLMProvider

```java
public interface LLMProvider {

    String id();

    ProviderCapabilities capabilities();

    InferenceResponse infer(ProviderRequest request);
}
```

### ProviderCapabilities

```java
public final class ProviderCapabilities {

    private final boolean streaming;
    private final boolean tools;
    private final boolean multimodal;
    private final int maxContextTokens;
}
```

---

## 8️⃣ Provider Dispatch (Normalized)

```java
public final class ProviderRequest {

    private final String model;
    private final List<Message> messages;
    private final Map<String, Object> parameters;
    private final boolean streaming;
}
```

* Mapped per provider
* Transport-agnostic

---

## 9️⃣ Streaming Support (Optional)

### StreamingLLMProvider

```java
public interface StreamingLLMProvider
        extends LLMProvider {

    StreamingResponse stream(ProviderRequest request);
}
```

### StreamingResponse

```java
public final class StreamingResponse
        implements InferenceResponse {

    private final Publisher<StreamChunk> publisher;
}
```

---

## 🔟 Observability (Hooks Only)

### InferenceObserver

```java
public interface InferenceObserver {

    void onStart(InferenceContext context);
    void onPhase(InferencePhase phase, InferenceContext context);
    void onSuccess(InferenceContext context);
    void onFailure(Throwable error, InferenceContext context);
}
```

* Metrics
* Tracing
* Logging

Kernel never logs directly.

---

## 1️⃣1️⃣ Safety & Policy (Plugins)

* Prompt validation
* Output moderation
* Policy enforcement
* Quotas / rate limits

All implemented as **InferencePhasePlugin**.

---

## 1️⃣2️⃣ What Is Explicitly NOT in the Kernel

❌ Workflow / BPMN
❌ Orchestration
❌ Agent runtime
❌ Long-running state
❌ Persistence
❌ Human-in-the-loop
❌ Business semantics

Those belong to **golek higher layers**, not here.

---

## 🧩 Final Kernel Boundary

```
golek-inference-kernel
├── engine
├── pipeline
├── phases
├── plugins
├── providers
├── streaming
├── observability
└── safety
```

This kernel is:

✔ coherent
✔ minimal
✔ extensible
✔ production-grade