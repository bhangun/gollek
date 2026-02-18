# Production Readiness & Future-Proofing Improvement Plan

## Executive Summary

The current foundation is solid but needs additional components to be **truly production-ready** and **future-proof**. This document outlines critical improvements across 12 categories.

---

## 1. Missing Critical Components

### 1.1 Exception Hierarchy & Error Handling

**Current State**: Basic exceptions mentioned but not implemented
**Required**:

```java
// inference-core-api/src/main/java/.../exception/

public class InferenceException extends RuntimeException {
    private final ErrorCode errorCode;
    private final Map<String, Object> context;
    // Proper exception chaining, context preservation
}

public enum ErrorCode {
    MODEL_NOT_FOUND(404, "MODEL_001"),
    TENSOR_SHAPE_MISMATCH(400, "TENSOR_001"),
    DEVICE_NOT_AVAILABLE(503, "DEVICE_001"),
    QUOTA_EXCEEDED(429, "QUOTA_001"),
    INITIALIZATION_FAILED(500, "INIT_001"),
    INFERENCE_TIMEOUT(504, "TIMEOUT_001"),
    // ... with HTTP status codes for API mapping
}

public class RetryableException extends InferenceException {
    private final RetryPolicy retryPolicy;
    // Distinguishes transient vs permanent failures
}

public class TenantException extends InferenceException {
    private final String requestId;
    // Tenant-specific errors for proper isolation
}
```

**Why Critical**: Proper exception handling is essential for:
- Client error vs server error distinction
- Automatic retry logic
- Observability (error classification in metrics)
- Debugging production issues

---

### 1.2 Complete REST/gRPC API Layer

**Current State**: No API implementation
**Required**:

```java
// inference-service-gateway/src/main/java/.../api/

@Path("/v1/infer")
@Produces(MediaType.APPLICATION_JSON)
@Consumes(MediaType.APPLICATION_JSON)
public class InferenceResource {
    
    @POST
    @Timed(value = "inference.request.duration")
    @Counted(value = "inference.request.total")
    @RolesAllowed({"USER", "ADMIN"})
    public Uni<InferenceResponse> infer(
        @HeaderParam("X-API-Key") String requestId,
        @HeaderParam("X-Request-ID") String requestId,
        @Valid InferenceRequest request) {
        
        // Async processing with Mutiny
        return inferenceService.processAsync(request)
            .onFailure().retry().atMost(3)
            .onFailure().invoke(e -> metrics.recordFailure(e));
    }
    
    @POST
    @Path("/stream")
    @Produces(MediaType.SERVER_SENT_EVENTS)
    public Multi<StreamingResponse> inferStream(...) {
        // For LLMs - token-by-token streaming
    }
}

// gRPC Service
@GrpcService
public class InferenceGrpcService implements InferenceGrpc.InferenceImplBase {
    @Override
    public void infer(InferenceProto.Request request, 
                      StreamObserver<InferenceProto.Response> responseObserver) {
        // Binary protocol for high-throughput scenarios
    }
}
```

**Additional Requirements**:
- OpenAPI 3.0 specification generation
- Request/response validation with Bean Validation 3.0
- Rate limiting per tenant (Redis-backed)
- API versioning strategy (v1, v2 URL prefixes)
- Content negotiation (JSON, Protocol Buffers, MessagePack)

---

### 1.3 Model Registry Service

**Current State**: ModelManifest POJO only
**Required**:

```java
// Full CRUD service for model lifecycle management

public interface ModelRegistryService {
    
    // Upload and register model
    CompletableFuture<ModelVersion> registerModel(
        String requestId,
        ModelUploadRequest upload);
    
    // Retrieve model manifest
    Optional<ModelManifest> getManifest(String modelId, String version);
    
    // List models with pagination
    Page<ModelSummary> listModels(String requestId, PageRequest page);
    
    // Model versioning
    void promoteVersion(String modelId, String version, Stage stage);
    void deprecateVersion(String modelId, String version);
    
    // Model conversion tracking
    ConversionJob requestConversion(String modelId, ModelFormat targetFormat);
    ConversionStatus getConversionStatus(String jobId);
}

@Entity
@Table(name = "models")
public class Model {
    @Id UUID id;
    String requestId;
    String name;
    String description;
    ModelFormat format;
    String storageUri;
    Map<String, String> metadata;
    
    @OneToMany
    List<ModelVersion> versions;
    
    @Enumerated
    Stage stage; // DEVELOPMENT, STAGING, PRODUCTION, DEPRECATED
    
    LocalDateTime createdAt;
    LocalDateTime updatedAt;
}
```

**Database Schema**:
```sql
CREATE TABLE models (
    id UUID PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    name VARCHAR(255) NOT NULL,
    format VARCHAR(50) NOT NULL,
    storage_uri TEXT NOT NULL,
    stage VARCHAR(50) NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL,
    UNIQUE(tenant_id, name)
);

CREATE TABLE model_versions (
    id UUID PRIMARY KEY,
    model_id UUID REFERENCES models(id),
    version VARCHAR(50) NOT NULL,
    checksum VARCHAR(64) NOT NULL,
    size_bytes BIGINT NOT NULL,
    manifest JSONB NOT NULL,
    created_at TIMESTAMP NOT NULL,
    UNIQUE(model_id, version)
);

CREATE INDEX idx_models_tenant ON models(tenant_id);
CREATE INDEX idx_model_versions_model ON model_versions(model_id);
```

---

### 1.4 Actual Model Converter Implementation

**Current State**: Interface only
**Required**:

```java
public class LiteRTConverter implements ModelConverter {
    
    @Override
    public ConversionResult convert(ConversionRequest request) {
        // Use TensorFlow Python subprocess or native converter
        // 1. Load source model (PyTorch/ONNX/TF)
        // 2. Convert to TFLite format
        // 3. Apply quantization if requested
        // 4. Validate converted model
        // 5. Upload to storage
        // 6. Generate manifest
    }
}

// Quantization support
public interface QuantizationStrategy {
    byte[] quantize(byte[] modelData, QuantizationConfig config);
}

public class DynamicRangeQuantization implements QuantizationStrategy {
    // FP32 → INT8 with calibration dataset
}

public class Float16Quantization implements QuantizationStrategy {
    // FP32 → FP16 for GPU inference
}
```

**Conversion Pipeline**:
```
PyTorch → ONNX → TFLite (via tf.lite.OnnxConverter)
TensorFlow → TFLite (via tf.lite.TFLiteConverter)
ONNX → TFLite (via onnx-tf + tf.lite)
JAX → TFLite (via jax2tf)
```

**Why Critical**: Without this, users can't easily onboard models from other frameworks.

---

## 2. Security Hardening

### 2.1 Authentication & Authorization

**Current State**: None
**Required**:

```java
// JWT-based authentication
@ApplicationScoped
public class JwtAuthenticationMechanism {
    
    public SecurityIdentity authenticate(String token) {
        // Validate JWT signature
        // Extract claims (requestId, userId, roles)
        // Build SecurityIdentity
    }
}

// Fine-grained authorization
@RolesAllowed({"MODEL_ADMIN"})
public void deleteModel(String modelId) { ... }

@RequiresTenant
public InferenceResponse infer(InferenceRequest request) {
    // Automatic tenant context validation
}

// Resource-based access control
public interface ResourcePolicy {
    boolean canAccess(Principal principal, Resource resource, Action action);
}
```

**OAuth2/OIDC Integration**:
```yaml
quarkus:
  oidc:
    auth-server-url: https://auth.example.com/realms/inference
    client-id: inference-engine
    credentials:
      secret: ${OIDC_CLIENT_SECRET}
```

**Tenant Isolation**:
```java
@TenantFilter
public class RequestContextFilter implements ContainerRequestFilter {
    @Override
    public void filter(ContainerRequestContext ctx) {
        String requestId = ctx.getHeaderString("X-API-Key");
        // Validate tenant
        // Set ThreadLocal context
        RequestContext.set(requestId);
    }
}
```

---

### 2.2 Model Signing & Verification

**Current State**: None
**Required**:

```java
public interface ModelSignatureVerifier {
    
    void signModel(Path modelPath, PrivateKey signingKey);
    
    boolean verifyModel(Path modelPath, PublicKey publicKey) 
        throws SignatureException;
}

// Use EdDSA (Ed25519) for performance
public class Ed25519ModelSigner implements ModelSignatureVerifier {
    
    @Override
    public void signModel(Path modelPath, PrivateKey key) {
        byte[] modelBytes = Files.readAllBytes(modelPath);
        byte[] signature = signData(modelBytes, key);
        
        // Store signature in manifest or separate file
        Path sigPath = modelPath.resolveSibling(modelPath.getFileName() + ".sig");
        Files.write(sigPath, signature);
    }
}

// Automatic verification on model load
@ApplicationScoped
public class SecureModelLoader {
    
    public byte[] loadAndVerify(String modelUri, String requestId) {
        byte[] modelData = storage.download(modelUri);
        byte[] signature = storage.download(modelUri + ".sig");
        
        PublicKey publicKey = keyManager.getPublicKey(requestId);
        
        if (!verifier.verify(modelData, signature, publicKey)) {
            throw new SecurityException("Model signature verification failed");
        }
        
        return modelData;
    }
}
```

---

### 2.3 Secrets Management

**Current State**: Environment variables only
**Required**:

```yaml
# Vault integration
quarkus:
  vault:
    url: https://vault.example.com:8200
    authentication:
      kubernetes:
        role: inference-engine
    secret:
      config-kv:
        path: inference/config

# Usage
@ConfigProperty(name = "storage.s3.access-key-id", 
                 defaultValue = "vault://secret/s3/access-key")
String s3AccessKey;
```

**Encryption at Rest**:
```java
public class EncryptedModelStorage implements ModelStorage {
    
    private final AESGCMEncryption encryption;
    
    @Override
    public void store(String modelId, byte[] modelData) {
        byte[] encrypted = encryption.encrypt(modelData, getKey(modelId));
        underlyingStorage.store(modelId, encrypted);
    }
    
    @Override
    public byte[] load(String modelId) {
        byte[] encrypted = underlyingStorage.load(modelId);
        return encryption.decrypt(encrypted, getKey(modelId));
    }
}
```

---

## 3. Observability Enhancement

### 3.1 Structured Logging

**Current State**: Basic SLF4J
**Required**:

```java
@Slf4j
public class InferenceLogger {
    
    public void logInferenceRequest(InferenceRequest request, String runner) {
        log.info("Inference request", 
            kv("requestId", request.getRequestId()),
            kv("requestId", request.getRequestId()),
            kv("modelId", request.getModelId()),
            kv("runner", runner),
            kv("inputSizeBytes", calculateSize(request.getInputs())),
            kv("timestamp", Instant.now())
        );
    }
    
    public void logInferenceComplete(InferenceResponse response) {
        log.info("Inference complete",
            kv("requestId", response.getRequestId()),
            kv("latencyMs", response.getLatencyMs()),
            kv("runner", response.getRunnerName()),
            kv("outputSizeBytes", calculateSize(response.getOutputs()))
        );
    }
}

// JSON output format for log aggregation
{
  "timestamp": "2025-01-30T10:15:30Z",
  "level": "INFO",
  "message": "Inference complete",
  "requestId": "req-123",
  "requestId": "tenant-1",
  "latencyMs": 12,
  "runner": "litert-cpu",
  "traceId": "trace-456"
}
```

---

### 3.2 Comprehensive Metrics

**Current State**: Basic counters
**Required**:

```java
@ApplicationScoped
public class InferenceMetrics {
    
    private final MeterRegistry registry;
    
    // Counters
    private final Counter requestsTotal;
    private final Counter failuresTotal;
    private final Counter quotaExceeded;
    
    // Timers (with percentiles)
    private final Timer inferenceLatency;
    private final Timer modelLoadTime;
    private final Timer tensorConversionTime;
    
    // Gauges
    private final AtomicInteger activeRequests;
    private final AtomicLong gpuMemoryUsed;
    
    // Histograms
    private final DistributionSummary inputSizeBytes;
    private final DistributionSummary outputSizeBytes;
    
    public void recordInference(InferenceRequest req, InferenceResponse res) {
        requestsTotal.increment();
        
        inferenceLatency.record(res.getLatencyMs(), TimeUnit.MILLISECONDS);
        
        Tags tags = Tags.of(
            "tenant", req.getRequestId(),
            "model", req.getModelId(),
            "runner", res.getRunnerName(),
            "device", res.getDeviceType().getId()
        );
        
        registry.counter("inference.success", tags).increment();
    }
}
```

**Additional Metrics Needed**:
```
# Resource utilization
inference.runner.memory.used{runner, device}
inference.runner.memory.available{runner, device}
inference.gpu.utilization{gpu_id}
inference.tpu.utilization{tpu_id}

# Model-specific
inference.model.cache.hit{model}
inference.model.cache.miss{model}
inference.model.load.duration{model}

# Tenant-specific
inference.tenant.requests.total{tenant}
inference.tenant.quota.used{tenant, resource}
inference.tenant.quota.limit{tenant, resource}

# Queue metrics
inference.queue.size
inference.queue.wait.duration
```

---

### 3.3 Distributed Tracing Enhancement

**Current State**: Basic OpenTelemetry mention
**Required**:

```java
@ApplicationScoped
public class TracingInterceptor {
    
    @Inject Tracer tracer;
    
    public InferenceResponse traceInference(
            InferenceRequest request, 
            Supplier<InferenceResponse> execution) {
        
        Span span = tracer.spanBuilder("inference.execute")
            .setSpanKind(SpanKind.SERVER)
            .setAttribute("tenant.id", request.getRequestId())
            .setAttribute("model.id", request.getModelId())
            .setAttribute("request.id", request.getRequestId())
            .startSpan();
        
        try (Scope scope = span.makeCurrent()) {
            
            // Nested spans for detailed tracing
            Span routerSpan = tracer.spanBuilder("inference.router")
                .startSpan();
            // ... router selection
            routerSpan.end();
            
            Span runnerSpan = tracer.spanBuilder("inference.runner")
                .setAttribute("runner.name", selectedRunner)
                .startSpan();
            // ... actual inference
            runnerSpan.end();
            
            InferenceResponse response = execution.get();
            span.setStatus(StatusCode.OK);
            return response;
            
        } catch (Exception e) {
            span.recordException(e);
            span.setStatus(StatusCode.ERROR, e.getMessage());
            throw e;
        } finally {
            span.end();
        }
    }
}
```

---

## 4. Resilience Patterns

### 4.1 Circuit Breaker Implementation

**Current State**: Mentioned but not implemented
**Required**:

```java
@ApplicationScoped
public class ResilientModelRunner {
    
    private final Map<String, CircuitBreaker> circuitBreakers = new ConcurrentHashMap<>();
    
    @CircuitBreaker(
        requestVolumeThreshold = 20,
        failureRateThreshold = 50,
        delay = 5000,
        successThreshold = 2
    )
    public InferenceResponse inferWithCircuitBreaker(
            ModelRunner runner, 
            InferenceRequest request) {
        
        String key = runner.name();
        CircuitBreaker cb = circuitBreakers.computeIfAbsent(
            key, k -> CircuitBreaker.ofDefaults(k));
        
        return cb.executeSupplier(() -> runner.infer(request));
    }
}

// Circuit breaker states: CLOSED, OPEN, HALF_OPEN
// Metrics exposure
inference.circuit.breaker.state{runner} # 0=CLOSED, 1=OPEN, 2=HALF_OPEN
inference.circuit.breaker.failures{runner}
```

---

### 4.2 Bulkhead Pattern

**Current State**: None
**Required**:

```java
@ApplicationScoped
public class BulkheadIsolation {
    
    // Separate thread pools per runner to prevent resource exhaustion
    private final Map<String, ExecutorService> runnerExecutors;
    
    @Bulkhead(value = 10) // Max 10 concurrent requests per runner
    @Timeout(value = 30, unit = ChronoUnit.SECONDS)
    public CompletableFuture<InferenceResponse> executeIsolated(
            String runnerName,
            Callable<InferenceResponse> task) {
        
        ExecutorService executor = runnerExecutors.get(runnerName);
        return CompletableFuture.supplyAsync(() -> {
            try {
                return task.call();
            } catch (Exception e) {
                throw new CompletionException(e);
            }
        }, executor);
    }
}
```

---

### 4.3 Retry with Exponential Backoff

**Current State**: Basic retry mention
**Required**:

```java
public class ExponentialBackoffRetry {
    
    @Retry(
        maxAttempts = 3,
        delay = 100,
        maxDelay = 2000,
        jitter = 0.25,
        retryOn = {RetryableException.class, TimeoutException.class},
        abortOn = {ModelNotFoundException.class, QuotaExceededException.class}
    )
    public InferenceResponse inferWithRetry(
            ModelRunner runner,
            InferenceRequest request) {
        
        return runner.infer(request);
    }
}
```

---

## 5. Performance Optimization

### 5.1 Advanced Batching

**Current State**: Mentioned, not implemented
**Required**:

```java
@ApplicationScoped
public class DynamicBatcher {
    
    private final BlockingQueue<InferenceJob> queue = new LinkedBlockingQueue<>();
    private final ScheduledExecutorService scheduler;
    
    @PostConstruct
    public void init() {
        scheduler.scheduleAtFixedRate(this::processBatch, 0, 10, TimeUnit.MILLISECONDS);
    }
    
    public CompletableFuture<InferenceResponse> submitForBatching(
            InferenceRequest request) {
        
        CompletableFuture<InferenceResponse> future = new CompletableFuture<>();
        queue.offer(new InferenceJob(request, future));
        return future;
    }
    
    private void processBatch() {
        List<InferenceJob> batch = new ArrayList<>();
        queue.drainTo(batch, 32); // Max batch size
        
        if (batch.isEmpty()) return;
        
        // Combine into single batch request
        InferenceRequest batchRequest = combineToBatch(batch);
        
        // Execute batch inference
        InferenceResponse batchResponse = runner.infer(batchRequest);
        
        // Split results and complete futures
        splitAndComplete(batch, batchResponse);
    }
}
```

---

### 5.2 Model Caching Strategy

**Current State**: Warm pool basics
**Required**:

```java
@ApplicationScoped
public class ModelCache {
    
    // LRU cache with size-based eviction
    private final Cache<ModelKey, LoadedModel> cache;
    
    // Caffeine cache with weight-based eviction (by memory)
    public ModelCache() {
        this.cache = Caffeine.newBuilder()
            .maximumWeight(10_000_000_000L) // 10GB
            .weigher((ModelKey key, LoadedModel model) -> model.getMemoryBytes())
            .expireAfterAccess(Duration.ofHours(1))
            .recordStats()
            .build();
    }
    
    public LoadedModel getOrLoad(ModelManifest manifest) {
        return cache.get(
            new ModelKey(manifest.getName(), manifest.getVersion()),
            key -> loadModel(manifest)
        );
    }
    
    // Prewarming on startup
    @Startup
    public void prewarm() {
        List<String> popularModels = config.getPrewarmModels();
        popularModels.forEach(modelId -> {
            ModelManifest manifest = registry.getManifest(modelId);
            getOrLoad(manifest);
        });
    }
}
```

---

### 5.3 Zero-Copy Tensor Operations

**Current State**: Byte array copies
**Required**:

```java
public class ZeroCopyTensorConverter {
    
    // Use MemorySegment for zero-copy tensor operations
    public MemorySegment convertToNative(TensorData tensor, Arena arena) {
        
        // Allocate native memory
        MemorySegment nativeSegment = arena.allocate(tensor.getData().length);
        
        // Direct copy without intermediate buffer
        MemorySegment.copy(
            MemorySegment.ofArray(tensor.getData()), 0,
            nativeSegment, 0,
            tensor.getData().length
        );
        
        return nativeSegment;
    }
    
    // Map file directly to memory (for large models)
    public MemorySegment mapModelFile(Path modelPath) throws IOException {
        FileChannel channel = FileChannel.open(modelPath, StandardOpenOption.READ);
        return channel.map(
            FileChannel.MapMode.READ_ONLY, 
            0, 
            channel.size(),
            Arena.global()
        );
    }
}
```

---

## 6. Data Management

### 6.1 Database Layer

**Current State**: No persistence
**Required**:

```java
// Use Panache for simplified data access
@Entity
@Table(name = "inference_requests")
public class InferenceRequestEntity extends PanacheEntityBase {
    
    @Id
    @GeneratedValue
    public UUID id;
    
    public String requestId;
    public String modelId;
    public String requestId;
    
    @Enumerated(EnumType.STRING)
    public RequestStatus status;
    
    public Long latencyMs;
    public String runnerName;
    
    @Column(columnDefinition = "jsonb")
    public String metadata;
    
    public LocalDateTime createdAt;
    public LocalDateTime completedAt;
    
    // Panache query methods
    public static List<InferenceRequestEntity> findByTenant(String requestId) {
        return list("requestId", requestId);
    }
    
    public static Long countByTenantAndTimeRange(
            String requestId, 
            LocalDateTime start, 
            LocalDateTime end) {
        return count("requestId = ?1 and createdAt between ?2 and ?3", 
            requestId, start, end);
    }
}

// Flyway migration
-- V1__initial_schema.sql
CREATE TABLE inference_requests (
    id UUID PRIMARY KEY,
    tenant_id VARCHAR(255) NOT NULL,
    model_id VARCHAR(255) NOT NULL,
    request_id VARCHAR(255) NOT NULL,
    status VARCHAR(50) NOT NULL,
    latency_ms BIGINT,
    runner_name VARCHAR(100),
    metadata JSONB,
    created_at TIMESTAMP NOT NULL,
    completed_at TIMESTAMP,
    INDEX idx_tenant_created (tenant_id, created_at),
    INDEX idx_request_id (request_id)
);
```

---

### 6.2 Caching Layer (Redis)

**Current State**: None
**Required**:

```java
@ApplicationScoped
public class RedisCacheManager {
    
    @Inject RedisClient redis;
    
    // Cache model manifests
    public Optional<ModelManifest> getManifest(String modelId) {
        String key = "manifest:" + modelId;
        String json = redis.get(key);
        return Optional.ofNullable(json)
            .map(j -> objectMapper.readValue(j, ModelManifest.class));
    }
    
    // Cache tenant quotas
    public void incrementQuotaUsage(String requestId, String resource) {
        String key = "quota:" + requestId + ":" + resource;
        redis.incr(key);
        redis.expire(key, 3600); // Reset hourly
    }
    
    // Distributed locking for model loading
    public boolean acquireLock(String modelId, Duration timeout) {
        String key = "lock:model:" + modelId;
        return redis.setnx(key, "1", timeout.toSeconds());
    }
}
```

---

## 7. Testing Infrastructure

### 7.1 Unit Tests

**Current State**: Basic structure
**Required**:

```java
@QuarkusTest
public class LiteRTCpuRunnerTest {
    
    @Inject LiteRTCpuRunner runner;
    
    private ModelManifest testManifest;
    private RunnerConfiguration testConfig;
    
    @BeforeEach
    public void setup() throws IOException {
        // Copy test model from resources
        Path modelPath = copyTestModel("mobilenet_v2_test.tflite");
        
        testManifest = ModelManifest.builder()
            .name("mobilenet-v2-test")
            .version("1.0")
            .artifacts(Map.of("litert", modelPath.toString()))
            .build();
        
        testConfig = RunnerConfiguration.builder()
            .parameter("numThreads", 2)
            .build();
    }
    
    @Test
    public void testInitialization() {
        assertDoesNotThrow(() -> runner.initialize(testManifest, testConfig));
        assertTrue(runner.health());
    }
    
    @Test
    public void testInferenceWithRealModel() {
        runner.initialize(testManifest, testConfig);
        
        // Create 224x224x3 random image tensor
        float[] imageData = createRandomImageData();
        TensorData inputTensor = TensorData.fromFloatArray(
            imageData, 1, 224, 224, 3);
        
        InferenceRequest request = InferenceRequest.builder()
            .modelId("mobilenet-v2-test:1.0")
            .requestId("test-tenant")
            .addInput("input", inputTensor)
            .build();
        
        InferenceResponse response = runner.infer(request);
        
        assertNotNull(response);
        assertTrue(response.getLatencyMs() > 0);
        assertNotNull(response.getOutputs().get("MobilenetV2/Predictions/Reshape_1"));
        
        // Validate output shape [1, 1001] for ImageNet
        TensorData output = response.getOutputs().values().iterator().next();
        assertArrayEquals(new long[]{1, 1001}, output.getShape());
    }
    
    @Test
    public void testConcurrentInference() throws InterruptedException {
        runner.initialize(testManifest, testConfig);
        
        ExecutorService executor = Executors.newFixedThreadPool(10);
        CountDownLatch latch = new CountDownLatch(100);
        AtomicInteger successCount = new AtomicInteger(0);
        
        for (int i = 0; i < 100; i++) {
            executor.submit(() -> {
                try {
                    InferenceRequest request = createTestRequest();
                    InferenceResponse response = runner.infer(request);
                    assertNotNull(response);
                    successCount.incrementAndGet();
                } finally {
                    latch.countDown();
                }
            });
        }
        
        latch.await(30, TimeUnit.SECONDS);
        assertEquals(100, successCount.get());
    }
    
    @Test
    public void testMemoryLeaks() {
        runner.initialize(testManifest, testConfig);
        
        long initialMemory = getUsedMemory();
        
        // Run 1000 inferences
        for (int i = 0; i < 1000; i++) {
            InferenceRequest request = createTestRequest();
            runner.infer(request);
        }
        
        System.gc();
        long finalMemory = getUsedMemory();
        
        // Memory growth should be minimal (< 50MB for 1000 requests)
        long memoryGrowth = finalMemory - initialMemory;
        assertTrue(memoryGrowth < 50_000_000, 
            "Memory leak detected: " + memoryGrowth + " bytes");
    }
    
    @AfterEach
    public void cleanup() {
        runner.close();
    }
}
```

---

### 7.2 Integration Tests

**Current State**: None
**Required**:

```java
@QuarkusTest
@QuarkusTestResource(PostgresContainerResource.class)
@QuarkusTestResource(RedisContainerResource.class)
@QuarkusTestResource(MinioContainerResource.class)
public class EndToEndIntegrationTest {
    
    @Inject ModelRegistryService registry;
    @Inject InferenceService inference;
    
    @Test
    public void testFullModelLifecycle() {
        // 1. Upload model
        ModelUploadRequest upload = new ModelUploadRequest(
            "test-tenant",
            "test-model",
            "1.0",
            modelBytes
        );
        ModelVersion version = registry.registerModel(upload).join();
        assertNotNull(version.getId());
        
        // 2. Request conversion
        ConversionJob job = registry.requestConversion(
            version.getModelId(), 
            ModelFormat.LITERT
        );
        
        // Wait for conversion
        await().atMost(Duration.ofMinutes(2))
            .until(() -> registry.getConversionStatus(job.getId()).isComplete());
        
        // 3. Run inference
        InferenceRequest request = InferenceRequest.builder()
            .modelId("test-model:1.0")
            .requestId("test-tenant")
            .addInput("input", createTestTensor())
            .build();
        
        InferenceResponse response = inference.infer(request).join();
        assertNotNull(response);
        assertTrue(response.getLatencyMs() > 0);
        
        // 4. Verify metrics
        Double avgLatency = registry.getMetrics(version.getModelId())
            .getAverageLatencyMs();
        assertNotNull(avgLatency);
    }
    
    @Test
    public void testMultiTenantIsolation() {
        // Verify tenant A cannot access tenant B's models
        ModelManifest tenantAModel = registry.getManifest(
            "tenant-a-model", "1.0", "tenant-a");
        assertNotNull(tenantAModel);
        
        assertThrows(UnauthorizedException.class, () ->
            registry.getManifest("tenant-a-model", "1.0", "tenant-b")
        );
    }
}
```

---

### 7.3 Performance/Load Tests

**Current State**: None
**Required**:

```java
// Using Gatling for load testing
public class InferenceLoadSimulation extends Simulation {
    
    HttpProtocolBuilder httpProtocol = http
        .baseUrl("http://localhost:8080")
        .acceptHeader("application/json");
    
    ScenarioBuilder scn = scenario("Inference Load Test")
        .exec(
            http("Infer Request")
                .post("/v1/infer")
                .header("X-API-Key", "load-test")
                .body(StringBody(session -> generateInferenceRequest()))
                .check(status().is(200))
                .check(jsonPath("$.latencyMs").lessThan(100))
        );
    
    {
        setUp(
            scn.injectOpen(
                rampUsersPerSec(1).to(100).during(Duration.ofMinutes(2)),
                constantUsersPerSec(100).during(Duration.ofMinutes(5))
            )
        ).protocols(httpProtocol)
         .assertions(
             global().responseTime().percentile(95).lt(150),
             global().successfulRequests().percent().gt(99.5)
         );
    }
}
```

---

## 8. Deployment & Operations

### 8.1 Helm Chart

**Current State**: Basic Kubernetes YAML
**Required**:

```yaml
# charts/inference-engine/values.yaml
replicaCount: 3

image:
  repository: inference-engine
  tag: latest
  pullPolicy: IfNotPresent

service:
  type: ClusterIP
  port: 80
  targetPort: 8080

resources:
  requests:
    memory: "2Gi"
    cpu: "1000m"
  limits:
    memory: "4Gi"
    cpu: "2000m"

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

# Model-specific node affinity
nodeSelector:
  inference.ai/gpu: "true"

tolerations:
  - key: "nvidia.com/gpu"
    operator: "Exists"
    effect: "NoSchedule"

# Persistent storage for model cache
persistence:
  enabled: true
  size: 100Gi
  storageClass: "fast-ssd"
```

---

### 8.2 GitOps Pipeline

**Current State**: Basic CI mentioned
**Required**:

```yaml
# .github/workflows/ci-cd.yaml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up JDK 21
        uses: actions/setup-java@v4
        with:
          java-version: '21'
          distribution: 'temurin'
      
      - name: Cache Maven packages
        uses: actions/cache@v4
        with:
          path: ~/.m2
          key: ${{ runner.os }}-m2-${{ hashFiles('**/pom.xml') }}
      
      - name: Run tests
        run: mvn clean verify
      
      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v4
        with:
          file: ./target/site/jacoco/jacoco.xml
  
  build-and-push:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    strategy:
      matrix:
        profile: [cpu, gpu, tpu]
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Build with Maven
        run: mvn clean package -P${{ matrix.profile }} -DskipTests
      
      - name: Build Docker image
        run: |
          docker build \
            -f Dockerfile.${{ matrix.profile }} \
            -t ${{ secrets.REGISTRY }}/inference-engine:${{ matrix.profile }}-${{ github.sha }} \
            .
      
      - name: Push to registry
        run: |
          echo ${{ secrets.REGISTRY_TOKEN }} | docker login -u ${{ secrets.REGISTRY_USER }} --password-stdin
          docker push ${{ secrets.REGISTRY }}/inference-engine:${{ matrix.profile }}-${{ github.sha }}
  
  deploy-staging:
    needs: build-and-push
    runs-on: ubuntu-latest
    environment: staging
    
    steps:
      - name: Deploy to staging
        run: |
          helm upgrade --install inference-engine \
            ./charts/inference-engine \
            --set image.tag=${{ github.sha }} \
            --namespace staging
```

---

### 8.3 Monitoring Stack

**Current State**: Basic metrics endpoint
**Required**:

```yaml
# Prometheus ServiceMonitor
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: inference-engine
spec:
  selector:
    matchLabels:
      app: inference-engine
  endpoints:
    - port: metrics
      interval: 15s
      path: /q/metrics

# Grafana Dashboard (JSON)
{
  "dashboard": {
    "title": "Inference Engine Metrics",
    "panels": [
      {
        "title": "Request Rate",
        "targets": [{
          "expr": "rate(inference_requests_total[5m])"
        }]
      },
      {
        "title": "Latency Percentiles",
        "targets": [
          {"expr": "histogram_quantile(0.50, inference_latency_seconds_bucket)"},
          {"expr": "histogram_quantile(0.95, inference_latency_seconds_bucket)"},
          {"expr": "histogram_quantile(0.99, inference_latency_seconds_bucket)"}
        ]
      },
      {
        "title": "Error Rate",
        "targets": [{
          "expr": "rate(inference_failures_total[5m])"
        }]
      },
      {
        "title": "GPU Memory Usage",
        "targets": [{
          "expr": "inference_runner_memory_used{device='cuda'}"
        }]
      }
    ]
  }
}

# Alerting Rules
groups:
  - name: inference_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(inference_failures_total[5m]) > 0.05
        for: 5m
        annotations:
          summary: "High error rate detected"
      
      - alert: HighLatency
        expr: histogram_quantile(0.95, inference_latency_seconds_bucket) > 0.5
        for: 10m
        annotations:
          summary: "P95 latency above 500ms"
      
      - alert: ModelNotHealthy
        expr: runner_health == 0
        for: 2m
        annotations:
          summary: "Model runner unhealthy"
```

---

## 9. Documentation

### 9.1 OpenAPI Specification

**Current State**: None
**Required**:

```yaml
# Auto-generated from Quarkus annotations
openapi: 3.0.3
info:
  title: Enterprise Inference Engine API
  version: 1.0.0
  description: Production ML inference platform

paths:
  /v1/infer:
    post:
      summary: Run inference
      operationId: infer
      tags:
        - Inference
      security:
        - BearerAuth: []
      parameters:
        - name: X-API-Key
          in: header
          required: true
          schema:
            type: string
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/InferenceRequest'
      responses:
        '200':
          description: Successful inference
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/InferenceResponse'
        '400':
          description: Invalid request
        '429':
          description: Quota exceeded
        '503':
          description: Service unavailable

components:
  securitySchemes:
    BearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT
```

---

### 9.2 Architecture Decision Records (ADRs)

**Current State**: None
**Required**:

```markdown
# ADR-001: Use Java FFM Instead of JNI

## Status
Accepted

## Context
Need native interop for TensorFlow Lite C API. Two options:
1. Traditional JNI
2. Java Foreign Function & Memory API (FFM)

## Decision
Use FFM (JEP 454, finalized in JDK 22)

## Consequences
### Positive
- 40% performance improvement over JNI
- Type-safe memory operations
- No native code compilation
- Automatic memory management with Arena
- Better debugging experience

### Negative
- Requires JDK 21+ (not an issue for new projects)
- Less mature ecosystem than JNI
- Team needs to learn FFM patterns

---

# ADR-002: Multi-Tenancy at Application Layer

## Status
Accepted

## Context
Need tenant isolation. Options:
1. Database-level (separate schemas)
2. Application-level (tenant context)
3. Infrastructure-level (separate deployments)

## Decision
Application-level with strong enforcement

## Rationale
- Better resource utilization
- Easier operations (single deployment)
- Lower cost
- Can enforce with database row-level security

---

# ADR-003: Build-Time Adapter Selection

## Status
Accepted

## Context
Different deployments need different ML frameworks.

## Decision
Use Maven profiles + @IfBuildProperty for conditional compilation

## Benefits
- Smaller deployment artifacts
- Reduced attack surface
- Better cold start times
- Clear separation of concerns
```

---

## 10. Migration Path

### 10.1 Phased Rollout Strategy

```
Phase 1 (Weeks 1-2): Core Infrastructure
✅ Complete exception hierarchy
✅ REST API implementation
✅ Model registry service
✅ Database schema
✅ Authentication/Authorization

Phase 2 (Weeks 3-4): Additional Adapters
✅ ONNX Runtime adapter (CPU + GPU)
✅ TensorFlow adapter
✅ Model converter service
✅ Unit tests (80%+ coverage)

Phase 3 (Weeks 5-6): Production Hardening
✅ Circuit breakers & resilience
✅ Comprehensive metrics
✅ Distributed tracing
✅ Integration tests
✅ Load tests

Phase 4 (Weeks 7-8): Operations
✅ Helm charts
✅ CI/CD pipeline
✅ Monitoring dashboards
✅ Alerting rules
✅ Documentation

Phase 5 (Week 9): Beta Deployment
✅ Deploy to staging
✅ Load testing
✅ Security audit
✅ Performance tuning

Phase 6 (Week 10): Production Release
✅ Canary deployment (10% traffic)
✅ Monitor metrics
✅ Full rollout
✅ Post-launch review
```

---

## 11. Cost Optimization

### 11.1 Resource Management

```java
@ApplicationScoped
public class CostOptimizedScheduler {
    
    // Use cheaper CPU for non-latency-sensitive requests
    public ModelRunner selectRunner(InferenceRequest request) {
        if (request.getPriority() < 7) {
            // Low priority = use CPU even if GPU available
            return cpuRunner;
        }
        
        // Check spot instance availability
        if (spotInstanceAvailable() && !request.isLatencyCritical()) {
            return spotGpuRunner;
        }
        
        return defaultRunner;
    }
    
    // Scale down during off-peak hours
    @Scheduled(cron = "0 0 22 * * ?") // 10 PM daily
    public void scaleDownOffPeak() {
        warmPool.reduceTo(minSize);
    }
    
    @Scheduled(cron = "0 0 6 * * ?") // 6 AM daily
    public void scaleUpPeak() {
        warmPool.expandTo(maxSize);
    }
}
```

---

## 12. Future-Proofing Strategies

### 12.1 Plugin Architecture

```java
// Allow third-party adapters via SPI
public interface ModelRunnerProvider {
    String getName();
    ModelRunner createRunner();
    boolean isAvailable();
}

@ApplicationScoped
public class PluginManager {
    
    private final ServiceLoader<ModelRunnerProvider> providers;
    
    @PostConstruct
    public void discoverPlugins() {
        providers = ServiceLoader.load(ModelRunnerProvider.class);
        
        for (ModelRunnerProvider provider : providers) {
            if (provider.isAvailable()) {
                registry.register(provider.getName(), provider.createRunner());
                log.info("Registered plugin: {}", provider.getName());
            }
        }
    }
}
```

---

### 12.2 Version Strategy

```java
// Support multiple API versions simultaneously
@Path("/v1/infer")
public class InferenceV1Resource { ... }

@Path("/v2/infer")
public class InferenceV2Resource {
    // New features:
    // - Streaming support
    // - Multi-model pipelines
    // - Enhanced error details
}

// Gradual migration with feature flags
if (featureFlags.isEnabled("api-v2", requestId)) {
    return v2Handler.process(request);
} else {
    return v1Handler.process(request);
}
```

---

### 12.3 Edge Computing Readiness

```java
// Lightweight edge deployment profile
@ApplicationScoped
@IfBuildProperty(name = "deployment.mode", stringValue = "edge")
public class EdgeOptimizedRouter {
    
    // Prefer on-device NPU over cloud GPU
    @Override
    public ModelRunner selectRunner(InferenceRequest request) {
        if (npuRunner.isAvailable() && model.supportsNpu()) {
            return npuRunner; // Lowest latency, privacy-preserving
        }
        
        if (networkLatency() > 50) {
            return localCpuRunner; // Network too slow for cloud
        }
        
        return cloudGpuRunner;
    }
}
```

---

## Summary: Critical Path to Production

### Must-Have (Weeks 1-4)
1. ✅ Exception hierarchy with proper error codes
2. ✅ REST API with validation
3. ✅ Model registry with database
4. ✅ Authentication & authorization
5. ✅ ONNX adapter (most universally compatible)
6. ✅ Basic observability (metrics + logs)

### Should-Have (Weeks 5-6)
7. ✅ Circuit breakers & resilience
8. ✅ Model converter service
9. ✅ Integration tests
10. ✅ Helm charts for Kubernetes

### Nice-to-Have (Weeks 7-8)
11. ✅ Advanced batching
12. ✅ Load testing
13. ✅ Comprehensive documentation
14. ✅ CI/CD pipeline

### Future Enhancements
15. ⏳ Edge deployment support
16. ⏳ Multi-model pipelines
17. ⏳ AutoML integration
18. ⏳ Model versioning strategies

---

## Conclusion

The current foundation is **solid** but needs **~8-10 weeks** of additional development for production readiness. Key priorities:

1. **Error handling** (critical)
2. **API layer** (critical)
3. **Persistence** (critical)
4. **Security** (critical)
5. **Observability** (critical)
6. **Testing** (important)
7. **Operations** (important)

With these improvements, the platform will be **enterprise-grade** and **future-proof** for the next 5+ years.
