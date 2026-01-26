



## Golek Inference Server Architecture Overview

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
golek-inference-server/
├── pom.xml                                    # Parent POM with profiles
├── README.md
├── ARCHITECTURE.md
├── docker/
│   ├── Dockerfile.platform-jvm
│   ├── Dockerfile.platform-native
│   ├── Dockerfile.portable-jvm
│   └── Dockerfile.portable-native
├── k8s/
│   ├── base/
│   ├── overlays/
│   └── helm/
│
├── inference-api/                             # 📦 API Contracts (shared by all)
│   ├── pom.xml
│   └── src/main/java/tech/kayys/golek/inference/api/
│       ├── InferenceRequest.java
│       ├── InferenceResponse.java
│       ├── StreamChunk.java
│       ├── ModelMetadata.java
│       ├── TenantContext.java
│       ├── ErrorPayload.java                  # ⚠️ Standardized error
│       └── AuditPayload.java                  # 📝 Audit events
│
├── inference-kernel/                          # 🧠 Core Kernel (shared)
│   ├── pom.xml
│   └── src/main/java/tech/kayys/golek/inference/kernel/
│       ├── engine/
│       │   ├── InferenceEngine.java           # Main entry point
│       │   ├── EngineContext.java             # Global engine state
│       │   ├── InferenceContext.java          # Request-scoped state
│       │   └── DefaultInferenceEngine.java
│       ├── pipeline/
│       │   ├── InferencePipeline.java         # Phase executor
│       │   ├── InferencePhase.java            # Ordered phases
│       │   └── DefaultInferencePipeline.java
│       ├── plugin/
│       │   ├── Plugin.java                    # Base plugin contract
│       │   ├── PluginDescriptor.java          # Plugin metadata
│       │   ├── PluginContext.java             # Plugin initialization ctx
│       │   ├── PluginRegistry.java            # Plugin management
│       │   ├── PluginLoader.java              # Discovery & loading
│       │   ├── ConfigurablePlugin.java        # Dynamic config support
│       │   ├── InferencePhasePlugin.java      # Phase-bound plugins
│       │   └── PluginLifecycle.java           # Lifecycle states
│       ├── provider/
│       │   ├── LLMProvider.java               # Provider SPI
│       │   ├── ProviderRequest.java           # Normalized request
│       │   ├── ProviderCapabilities.java      # Feature flags
│       │   ├── StreamingLLMProvider.java      # Streaming support
│       │   └── ProviderRegistry.java          # Provider management
│       ├── safety/
│       │   ├── SafetyPlugin.java              # Safety validation plugin
│       │   ├── PolicyPlugin.java              # Policy enforcement plugin
│       │   └── ContentModerator.java          # Content moderation
│       ├── observability/
│       │   ├── InferenceObserver.java         # Observer hook
│       │   ├── MetricsCollector.java          # Metrics facade
│       │   ├── TraceContext.java              # Tracing propagation
│       │   └── AuditLogger.java               # Audit event logger
│       └── exceptions/
│           ├── InferenceException.java
│           ├── PluginException.java
│           ├── ProviderException.java
│           └── ValidationException.java
│
├── inference-plugins-api/                     # 🔌 Plugin Abstraction
│   ├── pom.xml
│   └── src/main/java/tech/kayys/golek/inference/plugins/
│       ├── validation/
│       │   ├── ModelValidationPlugin.java     # Pre-flight validation
│       │   └── RequestValidationPlugin.java
│       ├── transformation/
│       │   ├── InputTransformPlugin.java      # Request mutation
│       │   └── OutputTransformPlugin.java     # Response mutation
│       ├── routing/
│       │   ├── RouterPlugin.java              # Custom routing logic
│       │   └── LoadBalancerPlugin.java        # Load distribution
│       ├── policy/
│       │   ├── QuotaPlugin.java               # Rate limit/quota
│       │   ├── AuthzPlugin.java               # Authorization
│       │   └── CompliancePlugin.java          # Regulatory checks
│       └── telemetry/
│           ├── TracingPlugin.java             # Distributed tracing
│           └── LoggingPlugin.java             # Structured logging
│
├── inference-providers-spi/                   # 🔧 Provider Adapters SPI
│   ├── pom.xml
│   └── src/main/java/tech/kayys/golek/inference/providers/
│       ├── local/
│       │   ├── LocalProviderAdapter.java      # Base for local models
│       │   └── ModelLoader.java               # Artifact loading
│       ├── cloud/
│       │   ├── CloudProviderAdapter.java      # Base for cloud APIs
│       │   └── RateLimitHandler.java          # API rate limiting
│       └── streaming/
│           ├── StreamingAdapter.java          # Streaming support
│           └── ChunkProcessor.java            # SSE/WebSocket handlers
│
├── inference-provider-gguf/                   # 🦙 llama.cpp Provider
│   ├── pom.xml
│   └── src/main/java/tech/kayys/golek/inference/providers/gguf/
│       ├── GGUFProvider.java
│       ├── LlamaCppBinding.java               # JNI/FFM binding
│       ├── GGUFModelLoader.java
│       ├── GGUFSessionManager.java            # Context management
│       └── resources/
│           └── META-INF/
│               ├── services/
│               │   └── tech.kayys.golek.inference.kernel.provider.LLMProvider
│               └── native-image/
│                   └── reflect-config.json
│
├── inference-provider-onnx/                   # 🧮 ONNX Provider
│   ├── pom.xml
│   └── src/main/java/tech/kayys/golek/inference/providers/onnx/
│       ├── ONNXProvider.java
│       ├── ONNXRuntimeSession.java
│       ├── ExecutionProviderSelector.java     # CPU/CUDA/TensorRT
│       └── TensorConverter.java
│
├── inference-provider-triton/                 # 🚀 Triton Provider
│   ├── pom.xml
│   └── src/main/java/tech/kayys/golek/inference/providers/triton/
│       ├── TritonProvider.java
│       ├── TritonGrpcClient.java
│       ├── TritonHttpClient.java
│       ├── ModelRepository.java               # Model loading
│       └── proto/                             # Generated gRPC stubs
│
├── inference-provider-openai/                 # 🌐 OpenAI Provider
│   ├── pom.xml
│   └── src/main/java/tech/kayys/golek/inference/providers/openai/
│       ├── OpenAIProvider.java
│       ├── OpenAIClient.java
│       ├── StreamingHandler.java
│       └── FunctionCallMapper.java            # Tool calling
│
├── inference-provider-anthropic/              # 🤖 Anthropic Provider
│   └── (similar structure)
│
├── inference-core/                            # 🏛️ Platform Core Services
│   ├── pom.xml
│   └── src/main/java/tech/kayys/golek/inference/core/
│       ├── domain/
│       │   ├── ModelManifest.java             # Model metadata
│       │   ├── ModelVersion.java
│       │   ├── TenantId.java
│       │   ├── ResourceRequirements.java
│       │   └── InferenceSession.java
│       ├── service/
│       │   ├── InferenceOrchestrator.java     # Main orchestration
│       │   ├── ModelRouterService.java        # Intelligent routing
│       │   ├── SelectionPolicy.java           # Multi-factor scoring
│       │   ├── ModelRunnerFactory.java        # Warm pool manager
│       │   ├── FallbackStrategy.java          # Graceful degradation
│       │   └── CircuitBreakerManager.java     # Resilience patterns
│       ├── repository/
│       │   ├── ModelRepository.java           # Model metadata store
│       │   └── TenantConfigRepository.java    # Tenant settings
│       └── config/
│           ├── InferenceConfig.java
│           ├── TenantConfig.java
│           └── ProviderConfig.java
│
├── inference-infrastructure/                  # 🔩 Infrastructure Adapters
│   ├── pom.xml
│   └── src/main/java/tech/kayys/golek/inference/infrastructure/
│       ├── persistence/
│       │   ├── PostgresModelRepository.java   # PostgreSQL persistence
│       │   ├── S3ArtifactStore.java           # S3/MinIO storage
│       │   └── entity/
│       │       ├── ModelEntity.java
│       │       └── TenantEntity.java
│       ├── messaging/
│       │   ├── KafkaAuditPublisher.java       # Event streaming
│       │   └── events/
│       │       ├── InferenceStarted.java
│       │       ├── InferenceCompleted.java
│       │       └── InferenceFailed.java
│       ├── security/
│       │   ├── JwtTenantResolver.java
│       │   ├── KeycloakIntegration.java
│       │   ├── TenantFilter.java
│       │   └── VaultSecretManager.java        # Vault integration
│       └── observability/
│           ├── PrometheusMetrics.java
│           ├── OpenTelemetryTracing.java
│           └── StructuredLogger.java
│
├── inference-platform-runtime/                # 🖥️ Platform Deployment
│   ├── pom.xml
│   └── src/main/
│       ├── java/tech/kayys/golek/inference/platform/
│       │   ├── rest/
│       │   │   ├── InferenceResource.java     # REST endpoints
│       │   │   ├── ModelManagementResource.java
│       │   │   ├── HealthResource.java
│       │   │   ├── PluginManagementResource.java
│       │   │   └── filters/
│       │   │       ├── RequestIdFilter.java
│       │   │       └── RateLimitFilter.java
│       │   ├── lifecycle/
│       │   │   ├── StartupInitializer.java    # Warm pool init
│       │   │   └── ShutdownHandler.java       # Graceful shutdown
│       │   └── config/
│       │       └── PlatformConfig.java
│       └── resources/
│           ├── application.yml
│           ├── application-dev.yml
│           ├── application-prod.yml
│           └── META-INF/
│               └── microprofile-config.properties
│
├── inference-portable-runtime/                # 📦 Portable Agent Runtime
│   ├── pom.xml                                # Minimal dependencies
│   └── src/main/
│       ├── java/tech/kayys/golek/inference/portable/
│       │   ├── PortableInferenceEngine.java   # Embedded engine
│       │   ├── LocalModelLoader.java          # Filesystem models
│       │   ├── SimplePluginRegistry.java      # Minimal plugin support
│       │   └── config/
│       │       └── PortableConfig.java        # File-based config
│       └── resources/
│           ├── application-portable.yml
│           └── models/                        # Bundled models (optional)
│
└── inference-tests/
    ├── pom.xml
    └── src/test/
        ├── java/
        │   ├── unit/
        │   ├── integration/
        │   │   ├── PluginLifecycleTest.java
        │   │   ├── MultiFormatTest.java
        │   │   └── TenantIsolationTest.java
        │   ├── performance/
        │   │   └── LoadTest.java
        │   └── e2e/
        │       └── EndToEndTest.java
        └── resources/
            └── fixtures/
                ├── models/
                └── requests/
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


----

Exsiting Code
---


### InferenceRequest.java

```java
package tech.kayys.golek.inference.api;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import org.jetbrains.annotations.Nullable;

import java.time.Duration;
import java.util.*;

/**
 * Immutable inference request.
 * Thread-safe and serializable.
 */
public final class InferenceRequest {

    @NotBlank
    private final String requestId;

    @NotBlank
    private final String model;

    @NotNull
    private final List<Message> messages;

    private final Map<String, Object> parameters;
    private final boolean streaming;
    
    @Nullable
    private final String preferredProvider;
    
    @Nullable
    private final Duration timeout;
    
    private final int priority;

    @JsonCreator
    public InferenceRequest(
        @JsonProperty("requestId") String requestId,
        @JsonProperty("model") String model,
        @JsonProperty("messages") List<Message> messages,
        @JsonProperty("parameters") Map<String, Object> parameters,
        @JsonProperty("streaming") boolean streaming,
        @JsonProperty("preferredProvider") String preferredProvider,
        @JsonProperty("timeout") Duration timeout,
        @JsonProperty("priority") int priority
    ) {
        this.requestId = Objects.requireNonNull(requestId, "requestId");
        this.model = Objects.requireNonNull(model, "model");
        this.messages = Collections.unmodifiableList(new ArrayList<>(
            Objects.requireNonNull(messages, "messages")
        ));
        this.parameters = parameters != null 
            ? Collections.unmodifiableMap(new HashMap<>(parameters))
            : Collections.emptyMap();
        this.streaming = streaming;
        this.preferredProvider = preferredProvider;
        this.timeout = timeout;
        this.priority = priority;
    }

    // Getters
    public String getRequestId() { return requestId; }
    public String getModel() { return model; }
    public List<Message> getMessages() { return messages; }
    public Map<String, Object> getParameters() { return parameters; }
    public boolean isStreaming() { return streaming; }
    public Optional<String> getPreferredProvider() { return Optional.ofNullable(preferredProvider); }
    public Optional<Duration> getTimeout() { return Optional.ofNullable(timeout); }
    public int getPriority() { return priority; }

    // Builder
    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String requestId = UUID.randomUUID().toString();
        private String model;
        private final List<Message> messages = new ArrayList<>();
        private final Map<String, Object> parameters = new HashMap<>();
        private boolean streaming = false;
        private String preferredProvider;
        private Duration timeout;
        private int priority = 5;

        public Builder requestId(String requestId) {
            this.requestId = requestId;
            return this;
        }

        public Builder model(String model) {
            this.model = model;
            return this;
        }

        public Builder message(Message message) {
            this.messages.add(message);
            return this;
        }

        public Builder messages(List<Message> messages) {
            this.messages.addAll(messages);
            return this;
        }

        public Builder parameter(String key, Object value) {
            this.parameters.put(key, value);
            return this;
        }

        public Builder parameters(Map<String, Object> parameters) {
            this.parameters.putAll(parameters);
            return this;
        }

        public Builder temperature(double temperature) {
            this.parameters.put("temperature", temperature);
            return this;
        }

        public Builder maxTokens(int maxTokens) {
            this.parameters.put("max_tokens", maxTokens);
            return this;
        }

        public Builder topP(double topP) {
            this.parameters.put("top_p", topP);
            return this;
        }

        public Builder streaming(boolean streaming) {
            this.streaming = streaming;
            return this;
        }

        public Builder preferredProvider(String provider) {
            this.preferredProvider = provider;
            return this;
        }

        public Builder timeout(Duration timeout) {
            this.timeout = timeout;
            return this;
        }

        public Builder priority(int priority) {
            this.priority = priority;
            return this;
        }

        public InferenceRequest build() {
            Objects.requireNonNull(model, "model is required");
            if (messages.isEmpty()) {
                throw new IllegalStateException("At least one message is required");
            }
            return new InferenceRequest(
                requestId, model, messages, parameters, streaming,
                preferredProvider, timeout, priority
            );
        }
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof InferenceRequest that)) return false;
        return streaming == that.streaming &&
               priority == that.priority &&
               requestId.equals(that.requestId) &&
               model.equals(that.model) &&
               messages.equals(that.messages);
    }

    @Override
    public int hashCode() {
        return Objects.hash(requestId, model, messages, streaming, priority);
    }

    @Override
    public String toString() {
        return "InferenceRequest{" +
               "requestId='" + requestId + '\'' +
               ", model='" + model + '\'' +
               ", messageCount=" + messages.size() +
               ", streaming=" + streaming +
               ", priority=" + priority +
               '}';
    }
}
```

### Message.java

```java
package tech.kayys.golek.inference.api;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;

import java.util.Objects;

/**
 * Immutable message in a conversation.
 */
public final class Message {

    public enum Role {
        SYSTEM,
        USER,
        ASSISTANT,
        FUNCTION,
        TOOL
    }

    @NotNull
    private final Role role;

    @NotBlank
    private final String content;

    @JsonCreator
    public Message(
        @JsonProperty("role") Role role,
        @JsonProperty("content") String content
    ) {
        this.role = Objects.requireNonNull(role, "role");
        this.content = Objects.requireNonNull(content, "content");
    }

    public Role getRole() {
        return role;
    }

    public String getContent() {
        return content;
    }

    // Factory methods
    public static Message system(String content) {
        return new Message(Role.SYSTEM, content);
    }

    public static Message user(String content) {
        return new Message(Role.USER, content);
    }

    public static Message assistant(String content) {
        return new Message(Role.ASSISTANT, content);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof Message message)) return false;
        return role == message.role && content.equals(message.content);
    }

    @Override
    public int hashCode() {
        return Objects.hash(role, content);
    }

    @Override
    public String toString() {
        return "Message{role=" + role + ", content='" + 
               (content.length() > 50 ? content.substring(0, 50) + "..." : content) + "'}";
    }
}
```

### InferenceResponse.java

```java
package tech.kayys.golek.inference.api;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;

import java.time.Instant;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Immutable inference response.
 */
public final class InferenceResponse {

    @NotBlank
    private final String requestId;

    @NotBlank
    private final String content;

    private final String model;
    private final int tokensUsed;
    private final long durationMs;
    
    @NotNull
    private final Instant timestamp;

    private final Map<String, Object> metadata;

    @JsonCreator
    public InferenceResponse(
        @JsonProperty("requestId") String requestId,
        @JsonProperty("content") String content,
        @JsonProperty("model") String model,
        @JsonProperty("tokensUsed") int tokensUsed,
        @JsonProperty("durationMs") long durationMs,
        @JsonProperty("timestamp") Instant timestamp,
        @JsonProperty("metadata") Map<String, Object> metadata
    ) {
        this.requestId = Objects.requireNonNull(requestId, "requestId");
        this.content = Objects.requireNonNull(content, "content");
        this.model = model;
        this.tokensUsed = tokensUsed;
        this.durationMs = durationMs;
        this.timestamp = timestamp != null ? timestamp : Instant.now();
        this.metadata = metadata != null
            ? Collections.unmodifiableMap(new HashMap<>(metadata))
            : Collections.emptyMap();
    }

    // Getters
    public String getRequestId() { return requestId; }
    public String getContent() { return content; }
    public String getModel() { return model; }
    public int getTokensUsed() { return tokensUsed; }
    public long getDurationMs() { return durationMs; }
    public Instant getTimestamp() { return timestamp; }
    public Map<String, Object> getMetadata() { return metadata; }

    // Builder
    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String requestId;
        private String content;
        private String model;
        private int tokensUsed;
        private long durationMs;
        private Instant timestamp = Instant.now();
        private final Map<String, Object> metadata = new HashMap<>();

        public Builder requestId(String requestId) {
            this.requestId = requestId;
            return this;
        }

        public Builder content(String content) {
            this.content = content;
            return this;
        }

        public Builder model(String model) {
            this.model = model;
            return this;
        }

        public Builder tokensUsed(int tokensUsed) {
            this.tokensUsed = tokensUsed;
            return this;
        }

        public Builder durationMs(long durationMs) {
            this.durationMs = durationMs;
            return this;
        }

        public Builder timestamp(Instant timestamp) {
            this.timestamp = timestamp;
            return this;
        }

        public Builder metadata(String key, Object value) {
            this.metadata.put(key, value);
            return this;
        }

        public Builder metadata(Map<String, Object> metadata) {
            this.metadata.putAll(metadata);
            return this;
        }

        public InferenceResponse build() {
            Objects.requireNonNull(requestId, "requestId is required");
            Objects.requireNonNull(content, "content is required");
            return new InferenceResponse(
                requestId, content, model, tokensUsed,
                durationMs, timestamp, metadata
            );
        }
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof InferenceResponse that)) return false;
        return requestId.equals(that.requestId) &&
               content.equals(that.content);
    }

    @Override
    public int hashCode() {
        return Objects.hash(requestId, content);
    }

    @Override
    public String toString() {
        return "InferenceResponse{" +
               "requestId='" + requestId + '\'' +
               ", model='" + model + '\'' +
               ", tokensUsed=" + tokensUsed +
               ", durationMs=" + durationMs +
               '}';
    }
}
```

### TenantContext.java

```java
package tech.kayys.golek.inference.api;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.validation.constraints.NotBlank;

import java.util.*;

/**
 * Immutable tenant context for multi-tenancy.
 */
public final class TenantContext {

    @NotBlank
    private final String tenantId;

    private final String userId;
    private final Set<String> roles;
    private final Map<String, String> attributes;

    @JsonCreator
    public TenantContext(
        @JsonProperty("tenantId") String tenantId,
        @JsonProperty("userId") String userId,
        @JsonProperty("roles") Set<String> roles,
        @JsonProperty("attributes") Map<String, String> attributes
    ) {
        this.tenantId = Objects.requireNonNull(tenantId, "tenantId");
        this.userId = userId;
        this.roles = roles != null
            ? Collections.unmodifiableSet(new HashSet<>(roles))
            : Collections.emptySet();
        this.attributes = attributes != null
            ? Collections.unmodifiableMap(new HashMap<>(attributes))
            : Collections.emptyMap();
    }

    public String getTenantId() {
        return tenantId;
    }

    public Optional<String> getUserId() {
        return Optional.ofNullable(userId);
    }

    public Set<String> getRoles() {
        return roles;
    }

    public boolean hasRole(String role) {
        return roles.contains(role);
    }

    public Map<String, String> getAttributes() {
        return attributes;
    }

    public Optional<String> getAttribute(String key) {
        return Optional.ofNullable(attributes.get(key));
    }

    // Factory method
    public static TenantContext of(String tenantId) {
        return new TenantContext(tenantId, null, null, null);
    }

    public static TenantContext of(String tenantId, String userId) {
        return new TenantContext(tenantId, userId, null, null);
    }

    // Builder
    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String tenantId;
        private String userId;
        private final Set<String> roles = new HashSet<>();
        private final Map<String, String> attributes = new HashMap<>();

        public Builder tenantId(String tenantId) {
            this.tenantId = tenantId;
            return this;
        }

        public Builder userId(String userId) {
            this.userId = userId;
            return this;
        }

        public Builder role(String role) {
            this.roles.add(role);
            return this;
        }

        public Builder roles(Set<String> roles) {
            this.roles.addAll(roles);
            return this;
        }

        public Builder attribute(String key, String value) {
            this.attributes.put(key, value);
            return this;
        }

        public Builder attributes(Map<String, String> attributes) {
            this.attributes.putAll(attributes);
            return this;
        }

        public TenantContext build() {
            Objects.requireNonNull(tenantId, "tenantId is required");
            return new TenantContext(tenantId, userId, roles, attributes);
        }
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof TenantContext that)) return false;
        return tenantId.equals(that.tenantId);
    }

    @Override
    public int hashCode() {
        return Objects.hash(tenantId);
    }

    @Override
    public String toString() {
        return "TenantContext{tenantId='" + tenantId + "', userId='" + userId + "'}";
    }
}
```

### ErrorPayload.java

```java
package tech.kayys.golek.inference.api;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;

import java.time.Instant;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Standardized error payload for all inference failures.
 * Integrates with golek's error-as-input pattern.
 */
public final class ErrorPayload {

    @NotBlank
    private final String type;

    @NotBlank
    private final String message;

    private final Map<String, Object> details;
    private final boolean retryable;
    private final String originNode;
    private final String originRunId;
    private final int attempt;
    private final int maxAttempts;

    @NotNull
    private final Instant timestamp;

    private final String suggestedAction;
    private final String provenanceRef;

    @JsonCreator
    public ErrorPayload(
        @JsonProperty("type") String type,
        @JsonProperty("message") String message,
        @JsonProperty("details") Map<String, Object> details,
        @JsonProperty("retryable") boolean retryable,
        @JsonProperty("originNode") String originNode,
        @JsonProperty("originRunId") String originRunId,
        @JsonProperty("attempt") int attempt,
        @JsonProperty("maxAttempts") int maxAttempts,
        @JsonProperty("timestamp") Instant timestamp,
        @JsonProperty("suggestedAction") String suggestedAction,
        @JsonProperty("provenanceRef") String provenanceRef
    ) {
        this.type = Objects.requireNonNull(type, "type");
        this.message = Objects.requireNonNull(message, "message");
        this.details = details != null
            ? Collections.unmodifiableMap(new HashMap<>(details))
            : Collections.emptyMap();
        this.retryable = retryable;
        this.originNode = originNode;
        this.originRunId = originRunId;
        this.attempt = attempt;
        this.maxAttempts = maxAttempts;
        this.timestamp = timestamp != null ? timestamp : Instant.now();
        this.suggestedAction = suggestedAction;
        this.provenanceRef = provenanceRef;
    }

    // Getters
    public String getType() { return type; }
    public String getMessage() { return message; }
    public Map<String, Object> getDetails() { return details; }
    public boolean isRetryable() { return retryable; }
    public String getOriginNode() { return originNode; }
    public String getOriginRunId() { return originRunId; }
    public int getAttempt() { return attempt; }
    public int getMaxAttempts() { return maxAttempts; }
    public Instant getTimestamp() { return timestamp; }
    public String getSuggestedAction() { return suggestedAction; }
    public String getProvenanceRef() { return provenanceRef; }

    // Factory methods
    public static ErrorPayload from(Throwable error, String nodeId, String runId) {
        return builder()
            .type(error.getClass().getSimpleName())
            .message(error.getMessage() != null ? error.getMessage() : "Unknown error")
            .originNode(nodeId)
            .originRunId(runId)
            .retryable(isRetryable(error))
            .suggestedAction(determineSuggestedAction(error))
            .detail("errorClass", error.getClass().getName())
            .detail("cause", error.getCause() != null ? error.getCause().getMessage() : null)
            .build();
    }

    private static boolean isRetryable(Throwable error) {
        String className = error.getClass().getName();
        return !className.contains("Validation") &&
               !className.contains("Authorization") &&
               !className.contains("Quota");
    }

    private static String determineSuggestedAction(Throwable error) {
        String className = error.getClass().getName();
        if (className.contains("Quota")) {
            return "escalate";
        } else if (className.contains("Provider")) {
            return "retry";
        } else if (className.contains("Validation")) {
            return "human_review";
        } else {
            return "fallback";
        }
    }

    // Builder
    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String type;
        private String message;
        private final Map<String, Object> details = new HashMap<>();
        private boolean retryable = false;
        private String originNode;
        private String originRunId;
        private int attempt = 0;
        private int maxAttempts = 3;
        private Instant timestamp = Instant.now();
        private String suggestedAction;
        private String provenanceRef;

        public Builder type(String type) {
            this.type = type;
            return this;
        }

        public Builder message(String message) {
            this.message = message;
            return this;
        }

        public Builder detail(String key, Object value) {
            if (value != null) {
                this.details.put(key, value);
            }
            return this;
        }

        public Builder details(Map<String, Object> details) {
            this.details.putAll(details);
            return this;
        }

        public Builder retryable(boolean retryable) {
            this.retryable = retryable;
            return this;
        }

        public Builder originNode(String originNode) {
            this.originNode = originNode;
            return this;
        }

        public Builder originRunId(String originRunId) {
            this.originRunId = originRunId;
            return this;
        }

        public Builder attempt(int attempt) {
            this.attempt = attempt;
            return this;
        }

        public Builder maxAttempts(int maxAttempts) {
            this.maxAttempts = maxAttempts;
            return this;
        }

        public Builder timestamp(Instant timestamp) {
            this.timestamp = timestamp;
            return this;
        }

        public Builder suggestedAction(String suggestedAction) {
            this.suggestedAction = suggestedAction;
            return this;
        }

        public Builder provenanceRef(String provenanceRef) {
            this.provenanceRef = provenanceRef;
            return this;
        }

        public ErrorPayload build() {
            Objects.requireNonNull(type, "type is required");
            Objects.requireNonNull(message, "message is required");
            return new ErrorPayload(
                type, message, details, retryable, originNode, originRunId,
                attempt, maxAttempts, timestamp, suggestedAction, provenanceRef
            );
        }
    }

    @Override
    public String toString() {
        return "ErrorPayload{" +
               "type='" + type + '\'' +
               ", message='" + message + '\'' +
               ", retryable=" + retryable +
               ", attempt=" + attempt +
               ", suggestedAction='" + suggestedAction + '\'' +
               '}';
    }
}
```

### AuditPayload.java

```java
package tech.kayys.golek.inference.api;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;
import org.apache.commons.codec.digest.DigestUtils;

import java.time.Instant;
import java.util.*;

/**
 * Audit event for inference operations.
 * Immutable and tamper-evident with hash.
 */
public final class AuditPayload {

    @NotNull
    private final Instant timestamp;

    @NotBlank
    private final String runId;

    private final String nodeId;

    @NotNull
    private final Actor actor;

    @NotBlank
    private final String event;

    @NotBlank
    private final String level;

    private final List<String> tags;
    private final Map<String, Object> metadata;
    private final Map<String, Object> contextSnapshot;

    @NotBlank
    private final String hash;

    @JsonCreator
    public AuditPayload(
        @JsonProperty("timestamp") Instant timestamp,
        @JsonProperty("runId") String runId,
        @JsonProperty("nodeId") String nodeId,
        @JsonProperty("actor") Actor actor,
        @JsonProperty("event") String event,
        @JsonProperty("level") String level,
        @JsonProperty("tags") List<String> tags,
        @JsonProperty("metadata") Map<String, Object> metadata,
        @JsonProperty("contextSnapshot") Map<String, Object> contextSnapshot,
        @JsonProperty("hash") String hash
    ) {
        this.timestamp = timestamp != null ? timestamp : Instant.now();
        this.runId = Objects.requireNonNull(runId, "runId");
        this.nodeId = nodeId;
        this.actor = Objects.requireNonNull(actor, "actor");
        this.event = Objects.requireNonNull(event, "event");
        this.level = Objects.requireNonNull(level, "level");
        this.tags = tags != null
            ? Collections.unmodifiableList(new ArrayList<>(tags))
            : Collections.emptyList();
        this.metadata = metadata != null
            ? Collections.unmodifiableMap(new HashMap<>(metadata))
            : Collections.emptyMap();
        this.contextSnapshot = contextSnapshot != null
            ? Collections.unmodifiableMap(new HashMap<>(contextSnapshot))
            : Collections.emptyMap();
        this.hash = Objects.requireNonNull(hash, "hash");
    }

    // Getters
    public Instant getTimestamp() { return timestamp; }
    public String getRunId() { return runId; }
    public String getNodeId() { return nodeId; }
    public Actor getActor() { return actor; }
    public String getEvent() { return event; }
    public String getLevel() { return level; }
    public List<String> getTags() { return tags; }
    public Map<String, Object> getMetadata() { return metadata; }
    public Map<String, Object> getContextSnapshot() { return contextSnapshot; }
    public String getHash() { return hash; }

    // Actor record
    public record Actor(
        @JsonProperty("type") String type, // system|human|agent
        @JsonProperty("id") String id,
        @JsonProperty("role") String role
    ) {
        public Actor {
            Objects.requireNonNull(type, "type");
            Objects.requireNonNull(id, "id");
        }

        public static Actor system(String id) {
            return new Actor("system", id, "system");
        }

        public static Actor human(String id, String role) {
            return new Actor("human", id, role);
        }

        public static Actor agent(String id) {
            return new Actor("agent", id, "agent");
        }
    }

    // Builder
    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private Instant timestamp = Instant.now();
        private String runId;
        private String nodeId;
        private Actor actor = Actor.system("inference-engine");
        private String event;
        private String level = "INFO";
        private final List<String> tags = new ArrayList<>();
        private final Map<String, Object> metadata = new HashMap<>();
        private final Map<String, Object> contextSnapshot = new HashMap<>();

        public Builder timestamp(Instant timestamp) {
            this.timestamp = timestamp;
            return this;
        }

        public Builder runId(String runId) {
            this.runId = runId;
            return this;
        }

        public Builder nodeId(String nodeId) {
            this.nodeId = nodeId;
            return this;
        }

        public Builder actor(Actor actor) {
            this.actor = actor;
            return this;
        }

        public Builder event(String event) {
            this.event = event;
            return this;
        }

        public Builder level(String level) {
            this.level = level;
            return this;
        }

        public Builder tag(String tag) {
            this.tags.add(tag);
            return this;
        }

        public Builder tags(List<String> tags) {
            this.tags.addAll(tags);
            return this;
        }

        public Builder metadata(String key, Object value) {
            this.metadata.put(key, value);
            return this;
        }

        public Builder metadata(Map<String, Object> metadata) {
            this.metadata.putAll(metadata);
            return this;
        }

        public Builder contextSnapshot(Map<String, Object> snapshot) {
            this.contextSnapshot.putAll(snapshot);
            return this;
        }

        public AuditPayload build() {
            Objects.requireNonNull(runId, "runId is required");
            Objects.requireNonNull(event, "event is required");

            String hash = computeHash(
                timestamp, runId, nodeId, actor.id(), event
            );

            return new AuditPayload(
                timestamp, runId, nodeId, actor, event, level,
                tags, metadata, contextSnapshot, hash
            );
        }

        private String computeHash(
            Instant timestamp,
            String runId,
            String nodeId,
            String actorId,
            String event
        ) {
            String content = String.join("|",
                timestamp.toString(),
                runId,
                nodeId != null ? nodeId : "",
                actorId,
                event
            );
            return DigestUtils.sha256Hex(content);
        }
    }

    @Override
    public String toString() {
        return "AuditPayload{" +"timestamp=" + timestamp +
               ", event='" + event + '\'' +
               ", level='" + level + '\'' +
               ", runId='" + runId + '\'' +
               '}';
    }
}
```


## 🔄 Execution Package

### ExecutionStatus.java

```java
package tech.kayys.golek.inference.kernel.execution;

/**
 * Canonical execution states for inference requests.
 * These states are the single source of truth.
 */
public enum ExecutionStatus {

    /**
     * Request created, not yet started
     */
    CREATED("Created", false, false),
    
    /**
     * Actively executing through phases
     */
    RUNNING("Running", false, false),
    
    /**
     * Waiting for external event (HITL, async callback)
     */
    WAITING("Waiting", false, false),
    
    /**
     * Paused by policy or manual intervention
     */
    SUSPENDED("Suspended", false, false),
    
    /**
     * In retry backoff period
     */
    RETRYING("Retrying", false, false),
    
    /**
     * Successfully completed
     */
    COMPLETED("Completed", true, false),
    
    /**
     * Terminal failure (exhausted retries)
     */
    FAILED("Failed", true, true),
    
    /**
     * Rollback/compensation completed
     */
    COMPENSATED("Compensated", true, false),
    
    /**
     * Cancelled by user/system
     */
    CANCELLED("Cancelled", true, false);

    private final String displayName;
    private final boolean terminal;
    private final boolean error;

    ExecutionStatus(String displayName, boolean terminal, boolean error) {
        this.displayName = displayName;
        this.terminal = terminal;
        this.error = error;
    }

    public String getDisplayName() {
        return displayName;
    }

    public boolean isTerminal() {
        return terminal;
    }

    public boolean isError() {
        return error;
    }

    public boolean isActive() {
        return !terminal;
    }

    public boolean canTransitionTo(ExecutionStatus target) {
        if (this == target) {
            return true;
        }
        if (this.isTerminal()) {
            return false;
        }

        return switch (this) {
            case CREATED -> target == RUNNING || target == CANCELLED;
            case RUNNING -> target == WAITING || target == RETRYING ||
                          target == COMPLETED || target == FAILED ||
                          target == SUSPENDED || target == CANCELLED;
            case WAITING -> target == RUNNING || target == FAILED ||
                          target == CANCELLED;
            case SUSPENDED -> target == RUNNING || target == CANCELLED;
            case RETRYING -> target == RUNNING || target == FAILED;
            default -> false;
        };
    }
}
```

### ExecutionSignal.java

```java
package tech.kayys.golek.inference.kernel.execution;

/**
 * Signals that trigger state transitions.
 * These are events, not states.
 */
public enum ExecutionSignal {

    /**
     * Execution started
     */
    START,
    
    /**
     * Phase completed successfully
     */
    PHASE_SUCCESS,
    
    /**
     * Phase failed
     */
    PHASE_FAILURE,
    
    /**
     * All phases completed
     */
    EXECUTION_SUCCESS,
    
    /**
     * Execution failed
     */
    EXECUTION_FAILURE,
    
    /**
     * Retry limit exhausted
     */
    RETRY_EXHAUSTED,
    
    /**
     * External wait requested (HITL, async)
     */
    WAIT_REQUESTED,
    
    /**
     * External approval received
     */
    APPROVED,
    
    /**
     * External rejection received
     */
    REJECTED,
    
    /**
     * Compensation triggered
     */
    COMPENSATE,
    
    /**
     * Compensation completed
     */
    COMPENSATION_DONE,
    
    /**
     * Cancellation requested
     */
    CANCEL,
    
    /**
     * Resume from suspended state
     */
    RESUME
}
```

### ExecutionToken.java

```java
package tech.kayys.golek.inference.kernel.execution;

import tech.kayys.golek.inference.kernel.pipeline.InferencePhase;

import java.io.Serializable;
import java.time.Instant;
import java.util.Collections;
import java.util.Map;
import java.util.Objects;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Immutable execution token representing the current state
 * of an inference request execution.
 * 
 * This is the single source of truth for execution state.
 * Serializable, persistable, and rehydration-safe.
 */
public final class ExecutionToken implements Serializable {

    private static final long serialVersionUID = 1L;

    private final String executionId;
    private final String requestId;
    private final ExecutionStatus status;
    private final InferencePhase currentPhase;
    private final int attempt;
    private final Instant createdAt;
    private final Instant lastUpdated;
    private final Map<String, Object> variables;
    private final Map<String, Object> metadata;

    private ExecutionToken(Builder builder) {
        this.executionId = builder.executionId;
        this.requestId = builder.requestId;
        this.status = builder.status;
        this.currentPhase = builder.currentPhase;
        this.attempt = builder.attempt;
        this.createdAt = builder.createdAt;
        this.lastUpdated = builder.lastUpdated;
        this.variables = new ConcurrentHashMap<>(builder.variables);
        this.metadata = new ConcurrentHashMap<>(builder.metadata);
    }

    // Getters
    public String getExecutionId() {
        return executionId;
    }

    public String getRequestId() {
        return requestId;
    }

    public ExecutionStatus getStatus() {
        return status;
    }

    public InferencePhase getCurrentPhase() {
        return currentPhase;
    }

    public int getAttempt() {
        return attempt;
    }

    public Instant getCreatedAt() {
        return createdAt;
    }

    public Instant getLastUpdated() {
        return lastUpdated;
    }

    public Map<String, Object> getVariables() {
        return Collections.unmodifiableMap(variables);
    }

    public Map<String, Object> getMetadata() {
        return Collections.unmodifiableMap(metadata);
    }

    /**
     * Create new token with updated status
     */
    public ExecutionToken withStatus(ExecutionStatus newStatus) {
        if (!status.canTransitionTo(newStatus)) {
            throw new IllegalStateTransitionException(
                "Cannot transition from " + status + " to " + newStatus
            );
        }
        return toBuilder()
            .status(newStatus)
            .lastUpdated(Instant.now())
            .build();
    }

    /**
     * Create new token with updated phase
     */
    public ExecutionToken withPhase(InferencePhase newPhase) {
        return toBuilder()
            .currentPhase(newPhase)
            .lastUpdated(Instant.now())
            .build();
    }

    /**
     * Create new token with incremented attempt
     */
    public ExecutionToken withNextAttempt() {
        return toBuilder()
            .attempt(attempt + 1)
            .lastUpdated(Instant.now())
            .build();
    }

    /**
     * Create new token with updated variable
     */
    public ExecutionToken withVariable(String key, Object value) {
        Map<String, Object> newVars = new ConcurrentHashMap<>(variables);
        newVars.put(key, value);
        return toBuilder()
            .variables(newVars)
            .lastUpdated(Instant.now())
            .build();
    }

    /**
     * Create new token with updated metadata
     */
    public ExecutionToken withMetadata(String key, Object value) {
        Map<String, Object> newMeta = new ConcurrentHashMap<>(metadata);
        newMeta.put(key, value);
        return toBuilder()
            .metadata(newMeta)
            .lastUpdated(Instant.now())
            .build();
    }

    public static Builder builder() {
        return new Builder();
    }

    public Builder toBuilder() {
        return new Builder()
            .executionId(executionId)
            .requestId(requestId)
            .status(status)
            .currentPhase(currentPhase)
            .attempt(attempt)
            .createdAt(createdAt)
            .lastUpdated(lastUpdated)
            .variables(variables)
            .metadata(metadata);
    }

    public static class Builder {
        private String executionId;
        private String requestId;
        private ExecutionStatus status = ExecutionStatus.CREATED;
        private InferencePhase currentPhase = InferencePhase.PRE_VALIDATE;
        private int attempt = 0;
        private Instant createdAt = Instant.now();
        private Instant lastUpdated = Instant.now();
        private Map<String, Object> variables = new ConcurrentHashMap<>();
        private Map<String, Object> metadata = new ConcurrentHashMap<>();

        public Builder executionId(String executionId) {
            this.executionId = executionId;
            return this;
        }

        public Builder requestId(String requestId) {
            this.requestId = requestId;
            return this;
        }

        public Builder status(ExecutionStatus status) {
            this.status = status;
            return this;
        }

        public Builder currentPhase(InferencePhase currentPhase) {
            this.currentPhase = currentPhase;
            return this;
        }

        public Builder attempt(int attempt) {
            this.attempt = attempt;
            return this;
        }

        public Builder createdAt(Instant createdAt) {
            this.createdAt = createdAt;
            return this;
        }

        public Builder lastUpdated(Instant lastUpdated) {
            this.lastUpdated = lastUpdated;
            return this;
        }

        public Builder variables(Map<String, Object> variables) {
            this.variables = new ConcurrentHashMap<>(variables);
            return this;
        }

        public Builder variable(String key, Object value) {
            this.variables.put(key, value);
            return this;
        }

        public Builder metadata(Map<String, Object> metadata) {
            this.metadata = new ConcurrentHashMap<>(metadata);
            return this;
        }

        public Builder metadataEntry(String key, Object value) {
            this.metadata.put(key, value);
            return this;
        }

        public ExecutionToken build() {
            if (executionId == null) {
                executionId = UUID.randomUUID().toString();
            }
            if (requestId == null) {
                throw new IllegalStateException("requestId is required");
            }
            return new ExecutionToken(this);
        }
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof ExecutionToken that)) return false;
        return executionId.equals(that.executionId);
    }

    @Override
    public int hashCode() {
        return Objects.hash(executionId);
    }

    @Override
    public String toString() {
        return "ExecutionToken{" +
               "executionId='" + executionId + '\'' +
               ", status=" + status +
               ", phase=" + currentPhase +
               ", attempt=" + attempt +
               '}';
    }
}
```

### IllegalStateTransitionException.java

```java
package tech.kayys.golek.inference.kernel.execution;

/**
 * Exception thrown when an illegal state transition is attempted.
 */
public class IllegalStateTransitionException extends RuntimeException {

    public IllegalStateTransitionException(String message) {
        super(message);
    }

    public IllegalStateTransitionException(String message, Throwable cause) {
        super(message, cause);
    }
}
```

### ExecutionStateMachine.java

```java
package tech.kayys.golek.inference.kernel.execution;

/**
 * Deterministic state machine for execution lifecycle.
 * Pure function: (current state, signal) -> next state
 */
public interface ExecutionStateMachine {

    /**
     * Compute next state based on current state and signal
     */
    ExecutionStatus next(ExecutionStatus current, ExecutionSignal signal);

    /**
     * Validate if transition is allowed
     */
    boolean isTransitionAllowed(ExecutionStatus from, ExecutionStatus to);
}
```

### DefaultExecutionStateMachine.java

```java
package tech.kayys.golek.inference.kernel.execution;

import jakarta.enterprise.context.ApplicationScoped;
import org.jboss.logging.Logger;

import java.util.Map;
import java.util.Set;

/**
 * Default implementation of execution state machine.
 * Enforces valid state transitions.
 */
@ApplicationScoped
public class DefaultExecutionStateMachine implements ExecutionStateMachine {

    private static final Logger LOG = Logger.getLogger(DefaultExecutionStateMachine.class);

    // Valid transitions map for validation
    private static final Map<ExecutionStatus, Set<ExecutionStatus>> ALLOWED_TRANSITIONS = Map.of(
        ExecutionStatus.CREATED, Set.of(
            ExecutionStatus.RUNNING, 
            ExecutionStatus.CANCELLED
        ),
        ExecutionStatus.RUNNING, Set.of(
            ExecutionStatus.WAITING, 
            ExecutionStatus.RETRYING,
            ExecutionStatus.COMPLETED, 
            ExecutionStatus.FAILED,
            ExecutionStatus.SUSPENDED, 
            ExecutionStatus.CANCELLED
        ),
        ExecutionStatus.WAITING, Set.of(
            ExecutionStatus.RUNNING, 
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED
        ),
        ExecutionStatus.SUSPENDED, Set.of(
            ExecutionStatus.RUNNING, 
            ExecutionStatus.CANCELLED
        ),
        ExecutionStatus.RETRYING, Set.of(
            ExecutionStatus.RUNNING, 
            ExecutionStatus.FAILED
        )
    );

    @Override
    public ExecutionStatus next(ExecutionStatus current, ExecutionSignal signal) {
        ExecutionStatus nextState = computeNextState(current, signal);
        
        if (!isTransitionAllowed(current, nextState)) {
            throw new IllegalStateTransitionException(
                String.format(
                    "Invalid transition from %s to %s via signal %s",
                    current, nextState, signal
                )
            );
        }
        
        LOG.debugf("State transition: %s -> %s (signal: %s)", current, nextState, signal);
        return nextState;
    }

    private ExecutionStatus computeNextState(ExecutionStatus current, ExecutionSignal signal) {
        return switch (current) {

            case CREATED -> switch (signal) {
                case START -> ExecutionStatus.RUNNING;
                case CANCEL -> ExecutionStatus.CANCELLED;
                default -> current;
            };

            case RUNNING -> switch (signal) {
                case EXECUTION_SUCCESS -> ExecutionStatus.COMPLETED;
                case PHASE_FAILURE, EXECUTION_FAILURE -> ExecutionStatus.RETRYING;
                case WAIT_REQUESTED -> ExecutionStatus.WAITING;
                case CANCEL -> ExecutionStatus.CANCELLED;
                case COMPENSATE -> ExecutionStatus.COMPENSATED;
                default -> current;
            };

            case RETRYING -> switch (signal) {
                case START, RESUME -> ExecutionStatus.RUNNING;
                case RETRY_EXHAUSTED -> ExecutionStatus.FAILED;
                case CANCEL -> ExecutionStatus.CANCELLED;
                default -> current;
            };

            case WAITING -> switch (signal) {
                case APPROVED, RESUME -> ExecutionStatus.RUNNING;
                case REJECTED -> ExecutionStatus.FAILED;
                case CANCEL -> ExecutionStatus.CANCELLED;
                default -> current;
            };

            case SUSPENDED -> switch (signal) {
                case RESUME -> ExecutionStatus.RUNNING;
                case CANCEL -> ExecutionStatus.CANCELLED;
                default -> current;
            };

            case COMPENSATED -> switch (signal) {
                case COMPENSATION_DONE -> ExecutionStatus.COMPLETED;
                default -> current;
            };

            // Terminal states - no transitions
            case COMPLETED, FAILED, CANCELLED -> current;
        };
    }

    @Override
    public boolean isTransitionAllowed(ExecutionStatus from, ExecutionStatus to) {
        // Self-transition always allowed
        if (from == to) {
            return true;
        }
        
        // Terminal states cannot transition
        if (from.isTerminal()) {
            return false;
        }
        
        Set<ExecutionStatus> allowed = ALLOWED_TRANSITIONS.get(from);
        return allowed != null && allowed.contains(to);
    }
}
```

### ExecutionContext.java

```java
package tech.kayys.golek.inference.kernel.execution;

import tech.kayys.golek.inference.api.TenantContext;
import tech.kayys.golek.inference.kernel.engine.EngineContext;
import tech.kayys.golek.inference.kernel.pipeline.InferencePhase;

import java.util.Map;
import java.util.Optional;

/**
 * Mutable execution context for a single inference request.
 * Provides access to execution token, variables, and engine context.
 */
public interface ExecutionContext {

    /**
     * Get engine context (global state)
     */
    EngineContext engine();

    /**
     * Get current execution token (immutable snapshot)
     */
    ExecutionToken token();

    /**
     * Get tenant context
     */
    TenantContext tenantContext();

    /**
     * Update execution status (creates new token)
     */
    void updateStatus(ExecutionStatus status);

    /**
     * Update current phase (creates new token)
     */
    void updatePhase(InferencePhase phase);

    /**
     * Increment retry attempt
     */
    void incrementAttempt();

    /**
     * Get execution variables (mutable view)
     */
    Map<String, Object> variables();

    /**
     * Put variable
     */
    void putVariable(String key, Object value);

    /**
     * Get variable
     */
    <T> Optional<T> getVariable(String key, Class<T> type);

    /**
     * Remove variable
     */
    void removeVariable(String key);

    /**
     * Get metadata
     */
    Map<String, Object> metadata();

    /**
     * Put metadata
     */
    void putMetadata(String key, Object value);

    /**
     * Replace entire token (for state restoration)
     */
    void replaceToken(ExecutionToken newToken);

    /**
     * Check if context has error
     */
    boolean hasError();

    /**
     * Get error if present
     */
    Optional<Throwable> getError();

    /**
     * Set error
     */
    void setError(Throwable error);

    /**
     * Clear error
     */
    void clearError();
}
```

### DefaultExecutionContext.java

```java
package tech.kayys.golek.inference.kernel.execution;

import tech.kayys.golek.inference.api.TenantContext;
import tech.kayys.golek.inference.kernel.engine.EngineContext;
import tech.kayys.golek.inference.kernel.pipeline.InferencePhase;

import java.util.Map;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Default implementation of execution context.
 */
public class DefaultExecutionContext implements ExecutionContext {

    private final EngineContext engineContext;
    private final TenantContext tenantContext;
    private final AtomicReference<ExecutionToken> tokenRef;
    private final AtomicReference<Throwable> errorRef;

    public DefaultExecutionContext(
        EngineContext engineContext,
        TenantContext tenantContext,
        ExecutionToken initialToken
    ) {
        this.engineContext = engineContext;
        this.tenantContext = tenantContext;
        this.tokenRef = new AtomicReference<>(initialToken);
        this.errorRef = new AtomicReference<>();
    }

    @Override
    public EngineContext engine() {
        return engineContext;
    }

    @Override
    public ExecutionToken token() {
        return tokenRef.get();
    }

    @Override
    public TenantContext tenantContext() {
        return tenantContext;
    }

    @Override
    public void updateStatus(ExecutionStatus status) {
        tokenRef.updateAndGet(token -> token.withStatus(status));
    }

    @Override
    public void updatePhase(InferencePhase phase) {
        tokenRef.updateAndGet(token -> token.withPhase(phase));
    }

    @Override
    public void incrementAttempt() {
        tokenRef.updateAndGet(ExecutionToken::withNextAttempt);
    }

    @Override
    public Map<String, Object> variables() {
        return token().getVariables();
    }

    @Override
    public void putVariable(String key, Object value) {
        tokenRef.updateAndGet(token -> token.withVariable(key, value));
    }

    @Override
    @SuppressWarnings("unchecked")
    public <T> Optional<T> getVariable(String key, Class<T> type) {
        Object value = variables().get(key);
        if (value == null) {
            return Optional.empty();
        }
        if (type.isInstance(value)) {
            return Optional.of((T) value);
        }
        return Optional.empty();
    }

    @Override
    public void removeVariable(String key) {
        Map<String, Object> vars = token().getVariables();
        vars.remove(key);
    }

    @Override
    public Map<String, Object> metadata() {
        return token().getMetadata();
    }

    @Override
    public void putMetadata(String key, Object value) {
        tokenRef.updateAndGet(token -> token.withMetadata(key, value));
    }

    @Override
    public void replaceToken(ExecutionToken newToken) {
        tokenRef.set(newToken);
    }

    @Override
    public boolean hasError() {
        return errorRef.get() != null;
    }

    @Override
    public Optional<Throwable> getError() {
        return Optional.ofNullable(errorRef.get());
    }

    @Override
    public void setError(Throwable error) {
        errorRef.set(error);
    }

    @Override
    public void clearError() {
        errorRef.set(null);
    }

    @Override
    public String toString() {
        return "DefaultExecutionContext{" +
               "token=" + token() +
               ", tenant=" + tenantContext.getTenantId() +
               ", hasError=" + hasError() +
               '}';
    }
}
```

---

## 🔄 Pipeline Package

### InferencePhase.java

```java
package tech.kayys.golek.inference.kernel.pipeline;

import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

/**
 * Comprehensive ordered phases of inference execution.
 * Each phase represents a distinct stage with specific responsibilities.
 * Plugins are bound to specific phases.
 */
public enum InferencePhase {
    
    /**
     * Phase 1: Pre-validation checks
     * - Request structure validation
     * - Basic sanity checks
     * - Early rejection of malformed requests
     */
    PRE_VALIDATE(1, "Pre-Validation"),
    
    /**
     * Phase 2: Deep validation
     * - Schema validation (JSON Schema)
     * - Content safety checks
     * - Input format verification
     * - Model compatibility checks
     */
    VALIDATE(2, "Validation"),
    
    /**
     * Phase 3: Authorization & access control
     * - Tenant verification
     * - Model access permissions
     * - Feature flag checks
     * - Quota verification
     */
    AUTHORIZE(3, "Authorization"),
    
    /**
     * Phase 4: Intelligent routing & provider selection
     * - Model-to-provider mapping
     * - Multi-factor scoring
     * - Load balancing
     * - Availability checks
     */
    ROUTE(4, "Routing"),
    
    /**
     * Phase 5: Request transformation & enrichment
     * - Prompt templating
     * - Context injection
     * - Parameter normalization
     * - Request mutation
     */
    PRE_PROCESSING(5, "Pre-Processing"),
    
    /**
     * Phase 6: Actual provider dispatch
     * - Provider invocation
     * - Streaming/batch execution
     * - Circuit breaker protection
     * - Fallback handling
     */
    PROVIDER_DISPATCH(6, "Provider Dispatch"),
    
    /**
     * Phase 7: Response post-processing
     * - Output validation
     * - Format normalization
     * - Metadata enrichment
     * - Content moderation
     */
    POST_PROCESSING(7, "Post-Processing"),
    
    /**
     * Phase 8: Audit logging
     * - Event recording
     * - Provenance tracking
     * - Compliance logging
     * - Immutable audit trail
     */
    AUDIT(8, "Audit"),
    
    /**
     * Phase 9: Observability & metrics
     * - Metrics emission
     * - Trace completion
     * - Performance tracking
     * - Cost accounting
     */
    OBSERVABILITY(9, "Observability"),
    
    /**
     * Phase 10: Resource cleanup
     * - Cache invalidation
     * - Connection release
     * - Quota release
     * - Temporary resource cleanup
     */
    CLEANUP(10, "Cleanup");

    private final int order;
    private final String displayName;

    InferencePhase(int order, String displayName) {
        this.order = order;
        this.displayName = displayName;
    }

    public int getOrder() {
        return order;
    }

    public String getDisplayName() {
        return displayName;
    }

    /**
     * Get phases in execution order
     */
    public static List<InferencePhase> ordered() {
        return Arrays.stream(values())
            .sorted(Comparator.comparing(InferencePhase::getOrder))
            .toList();
    }

    /**
     * Check if this phase is critical (execution cannot proceed if it fails)
     */
    public boolean isCritical() {
        return this == PRE_VALIDATE || 
               this == VALIDATE || 
               this == AUTHORIZE ||
               this == PROVIDER_DISPATCH;
    }

    /**
     * Check if this phase can be retried
     */
    public boolean isRetryable() {
        return this == ROUTE || 
               this == PROVIDER_DISPATCH;
    }

    /**
     * Check if this phase is idempotent
     */
    public boolean isIdempotent() {
        return this != PROVIDER_DISPATCH;
    }

    /**
     * Check if this phase should run even on error
     */
    public boolean runsOnError() {
        return this == AUDIT || 
               this == OBSERVABILITY || 
               this == CLEANUP;
    }
}
```

### InferencePipeline.java

```java
package tech.kayys.golek.inference.kernel.pipeline;

import tech.kayys.golek.inference.kernel.execution.ExecutionContext;
import io.smallrye.mutiny.Uni;

/**
 * Pipeline that executes all phases in order.
 */
public interface InferencePipeline {

    /**
     * Execute all phases for the given context
     */
    Uni<ExecutionContext> execute(ExecutionContext context);

    /**
     * Execute a specific phase
     */
    Uni<ExecutionContext> executePhase(ExecutionContext context, InferencePhase phase);
}
```
