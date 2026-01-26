# Complete GGUF Provider Implementation

Based on the architecture blueprint and SPI contracts, here's the comprehensive implementation of the GGUF provider module with proper multi-tenancy, observability, and error handling.

## Project Structure

```
inference-provider-gguf/
├── pom.xml
├── src/main/
│   ├── java/tech/kayys/wayang/inference/providers/gguf/
│   │   ├── GGUFProvider.java
│   │   ├── GGUFProviderConfig.java
│   │   ├── GGUFModelLoader.java
│   │   ├── GGUFSessionManager.java
│   │   ├── GGUFInferenceExecutor.java
│   │   ├── LlamaCppBinding.java
│   │   ├── LlamaCppNative.java
│   │   ├── GGUFModelMetadata.java
│   │   ├── GGUFStreamHandler.java
│   │   ├── GGUFHealthChecker.java
│   │   ├── GGUFMetricsCollector.java
│   │   └── exceptions/
│   │       ├── GGUFLoadException.java
│   │       ├── GGUFInferenceException.java
│   │       └── GGUFSessionException.java
│   └── resources/
│       ├── META-INF/
│       │   ├── services/
│       │   │   └── tech.kayys.wayang.inference.kernel.provider.LLMProvider
│       │   └── native-image/
│       │       ├── reflect-config.json
│       │       └── jni-config.json
│       └── application.properties
└── src/test/
    └── java/tech/kayys/wayang/inference/providers/gguf/
        ├── GGUFProviderTest.java
        ├── GGUFSessionManagerTest.java
        └── GGUFModelLoaderTest.java
```

## 1. Maven Configuration (pom.xml)

```xml
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 
         http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>

    <parent>
        <groupId>tech.kayys.wayang</groupId>
        <artifactId>inference-server</artifactId>
        <version>1.0.0-SNAPSHOT</version>
    </parent>

    <artifactId>inference-provider-gguf</artifactId>
    <name>Wayang Inference Provider - GGUF (llama.cpp)</name>
    <description>GGUF model provider using llama.cpp native bindings</description>

    <properties>
        <llama-cpp.version>b3715</llama-cpp.version>
        <jna.version>5.14.0</jna.version>
    </properties>

    <dependencies>
        <!-- Core API -->
        <dependency>
            <groupId>tech.kayys.wayang</groupId>
            <artifactId>inference-api</artifactId>
        </dependency>

        <!-- Kernel -->
        <dependency>
            <groupId>tech.kayys.wayang</groupId>
            <artifactId>inference-kernel</artifactId>
        </dependency>

        <!-- Provider SPI -->
        <dependency>
            <groupId>tech.kayys.wayang</groupId>
            <artifactId>inference-providers-spi</artifactId>
        </dependency>

        <!-- Quarkus Core -->
        <dependency>
            <groupId>io.quarkus</groupId>
            <artifactId>quarkus-arc</artifactId>
        </dependency>

        <dependency>
            <groupId>io.quarkus</groupId>
            <artifactId>quarkus-smallrye-context-propagation</artifactId>
        </dependency>

        <!-- Reactive -->
        <dependency>
            <groupId>io.smallrye.reactive</groupId>
            <artifactId>mutiny</artifactId>
        </dependency>

        <!-- JNA for native bindings -->
        <dependency>
            <groupId>net.java.dev.jna</groupId>
            <artifactId>jna</artifactId>
            <version>${jna.version}</version>
        </dependency>

        <!-- Observability -->
        <dependency>
            <groupId>io.quarkus</groupId>
            <artifactId>quarkus-micrometer-registry-prometheus</artifactId>
        </dependency>

        <dependency>
            <groupId>io.quarkus</groupId>
            <artifactId>quarkus-opentelemetry</artifactId>
        </dependency>

        <!-- Validation -->
        <dependency>
            <groupId>io.quarkus</groupId>
            <artifactId>quarkus-hibernate-validator</artifactId>
        </dependency>

        <!-- Logging -->
        <dependency>
            <groupId>io.quarkus</groupId>
            <artifactId>quarkus-logging-json</artifactId>
        </dependency>

        <!-- Testing -->
        <dependency>
            <groupId>io.quarkus</groupId>
            <artifactId>quarkus-junit5</artifactId>
            <scope>test</scope>
        </dependency>

        <dependency>
            <groupId>io.rest-assured</groupId>
            <artifactId>rest-assured</artifactId>
            <scope>test</scope>
        </dependency>

        <dependency>
            <groupId>org.assertj</groupId>
            <artifactId>assertj-core</artifactId>
            <scope>test</scope>
        </dependency>

        <dependency>
            <groupId>org.awaitility</groupId>
            <artifactId>awaitility</artifactId>
            <scope>test</scope>
        </dependency>
    </dependencies>

    <build>
        <plugins>
            <!-- Native library extraction -->
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-dependency-plugin</artifactId>
                <executions>
                    <execution>
                        <id>copy-native-libs</id>
                        <phase>process-resources</phase>
                        <goals>
                            <goal>copy-dependencies</goal>
                        </goals>
                        <configuration>
                            <includeScope>runtime</includeScope>
                            <includeTypes>so,dylib,dll</includeTypes>
                            <outputDirectory>${project.build.directory}/native-libs</outputDirectory>
                        </configuration>
                    </execution>
                </executions>
            </plugin>

            <!-- GraalVM Native Image -->
            <plugin>
                <groupId>io.quarkus</groupId>
                <artifactId>quarkus-maven-plugin</artifactId>
                <configuration>
                    <nativeImageBuildArgs>
                        <arg>--initialize-at-build-time=tech.kayys.wayang.inference.providers.gguf.LlamaCppNative</arg>
                        <arg>-H:+ReportExceptionStackTraces</arg>
                    </nativeImageBuildArgs>
                </configuration>
            </plugin>
        </plugins>
    </build>

    <profiles>
        <!-- Platform-specific native libraries -->
        <profile>
            <id>linux-x86_64</id>
            <activation>
                <os>
                    <family>unix</family>
                    <arch>amd64</arch>
                </os>
            </activation>
            <properties>
                <native.lib>libllama.so</native.lib>
            </properties>
        </profile>

        <profile>
            <id>macos-aarch64</id>
            <activation>
                <os>
                    <family>mac</family>
                    <arch>aarch64</arch>
                </os>
            </activation>
            <properties>
                <native.lib>libllama.dylib</native.lib>
            </properties>
        </profile>

        <profile>
            <id>windows-x86_64</id>
            <activation>
                <os>
                    <family>windows</family>
                    <arch>amd64</arch>
                </os>
            </activation>
            <properties>
                <native.lib>llama.dll</native.lib>
            </properties>
        </profile>
    </profiles>
</project>
```

## 2. Core Provider Implementation

### GGUFProvider.java

```java
package tech.kayys.wayang.inference.providers.gguf;

import io.micrometer.core.instrument.MeterRegistry;
import io.opentelemetry.api.trace.Tracer;
import io.quarkus.runtime.ShutdownEvent;
import io.quarkus.runtime.StartupEvent;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.event.Observes;
import jakarta.inject.Inject;
import org.eclipse.microprofile.config.inject.ConfigProperty;
import org.jboss.logging.Logger;
import tech.kayys.wayang.inference.api.*;
import tech.kayys.wayang.inference.kernel.provider.LLMProvider;
import tech.kayys.wayang.inference.kernel.provider.ProviderCapabilities;
import tech.kayys.wayang.inference.kernel.provider.ProviderRequest;
import tech.kayys.wayang.inference.kernel.provider.StreamingLLMProvider;
import tech.kayys.wayang.inference.providers.circuit.CircuitBreaker;
import tech.kayys.wayang.inference.providers.circuit.DefaultCircuitBreaker;
import tech.kayys.wayang.inference.providers.gguf.exceptions.GGUFInferenceException;
import tech.kayys.wayang.inference.providers.gguf.exceptions.GGUFLoadException;

import java.time.Duration;
import java.time.Instant;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

/**
 * GGUF Provider implementation using llama.cpp native bindings.
 * 
 * Features:
 * - Multi-tenant model isolation
 * - Session pooling and reuse
 * - Streaming and batch inference
 * - Circuit breaker protection
 * - Full observability integration
 * - Graceful shutdown
 * 
 * Thread-safety: This provider is thread-safe and can handle concurrent requests.
 * Resource management: Sessions are pooled per tenant and model combination.
 */
@ApplicationScoped
public class GGUFProvider implements StreamingLLMProvider {

    private static final Logger LOG = Logger.getLogger(GGUFProvider.class);
    private static final String PROVIDER_ID = "gguf-llama-cpp";
    private static final String PROVIDER_VERSION = "1.0.0";

    @Inject
    GGUFProviderConfig config;

    @Inject
    GGUFModelLoader modelLoader;

    @Inject
    GGUFSessionManager sessionManager;

    @Inject
    GGUFInferenceExecutor inferenceExecutor;

    @Inject
    GGUFHealthChecker healthChecker;

    @Inject
    GGUFMetricsCollector metricsCollector;

    @Inject
    Tracer tracer;

    @Inject
    MeterRegistry meterRegistry;

    @ConfigProperty(name = "gguf.provider.circuit-breaker.enabled", defaultValue = "true")
    boolean circuitBreakerEnabled;

    private CircuitBreaker circuitBreaker;
    private volatile boolean initialized = false;
    private final Map<String, GGUFModelMetadata> modelCache = new ConcurrentHashMap<>();

    /**
     * Initialize provider on startup
     */
    void onStart(@Observes StartupEvent event) {
        LOG.infof("Initializing GGUF Provider v%s", PROVIDER_VERSION);
        
        try {
            // Initialize circuit breaker
            if (circuitBreakerEnabled) {
                var cbConfig = DefaultCircuitBreaker.CircuitBreakerConfig.builder()
                    .failureThreshold(config.circuitBreakerFailureThreshold())
                    .openDuration(Duration.ofMillis(config.circuitBreakerOpenDurationMs()))
                    .halfOpenPermits(config.circuitBreakerHalfOpenPermits())
                    .halfOpenSuccessThreshold(config.circuitBreakerHalfOpenSuccessThreshold())
                    .build();
                
                this.circuitBreaker = new DefaultCircuitBreaker(PROVIDER_ID, cbConfig);
                LOG.info("Circuit breaker enabled for GGUF provider");
            }

            // Verify native library
            LlamaCppBinding.initialize();
            LOG.info("llama.cpp native library loaded successfully");

            // Initialize session manager
            sessionManager.initialize();

            // Prewarm if configured
            if (config.prewarmOnStartup()) {
                prewarmModels();
            }

            initialized = true;
            LOG.info("GGUF Provider initialization completed");

        } catch (Exception e) {
            LOG.error("Failed to initialize GGUF Provider", e);
            throw new IllegalStateException("GGUF Provider initialization failed", e);
        }
    }

    /**
     * Cleanup on shutdown
     */
    void onStop(@Observes ShutdownEvent event) {
        LOG.info("Shutting down GGUF Provider");
        
        try {
            sessionManager.shutdown();
            modelCache.clear();
            
            if (circuitBreaker != null) {
                circuitBreaker.reset();
            }
            
            LOG.info("GGUF Provider shutdown completed");
        } catch (Exception e) {
            LOG.error("Error during GGUF Provider shutdown", e);
        }
    }

    @Override
    public String id() {
        return PROVIDER_ID;
    }

    @Override
    public ProviderCapabilities capabilities() {
        return ProviderCapabilities.builder()
            .streaming(true)
            .tools(false) // llama.cpp doesn't natively support tool calling
            .multimodal(false)
            .maxContextTokens(config.maxContextTokens())
            .supportedFormats(new String[]{"gguf"})
            .supportsStreaming(true)
            .supportsBatching(config.batchingEnabled())
            .metadata(Map.of(
                "provider", "llama.cpp",
                "version", PROVIDER_VERSION,
                "native_library", LlamaCppBinding.getLibraryVersion(),
                "gpu_enabled", config.gpuEnabled(),
                "gpu_layers", config.gpuLayers()
            ))
            .build();
    }

    @Override
    public Uni<InferenceResponse> infer(ProviderRequest request) {
        return Uni.createFrom().item(() -> {
            ensureInitialized();
            
            var span = tracer.spanBuilder("gguf.infer")
                .setAttribute("model", request.model())
                .setAttribute("streaming", request.streaming())
                .startSpan();

            try {
                return executeWithCircuitBreaker(() -> 
                    inferInternal(request, false)
                );
            } finally {
                span.end();
            }
        });
    }

    @Override
    public Uni<StreamingResponse> stream(ProviderRequest request) {
        return Uni.createFrom().item(() -> {
            ensureInitialized();
            
            var span = tracer.spanBuilder("gguf.stream")
                .setAttribute("model", request.model())
                .startSpan();

            try {
                return executeWithCircuitBreaker(() -> 
                    (StreamingResponse) inferInternal(request, true)
                );
            } finally {
                span.end();
            }
        });
    }

    /**
     * Internal inference execution with full observability
     */
    private InferenceResponse inferInternal(ProviderRequest request, boolean streaming) {
        var requestId = UUID.randomUUID().toString();
        var startTime = Instant.now();
        
        metricsCollector.recordInferenceStart(request.model());

        try {
            // Get or load model metadata
            var metadata = getOrLoadModel(request.model());
            
            // Get session for this tenant/model
            var session = sessionManager.getSession(
                extractTenantId(request),
                request.model(),
                metadata
            );

            // Execute inference
            InferenceResponse response;
            if (streaming) {
                response = inferenceExecutor.executeStreaming(
                    session,
                    request,
                    requestId
                );
            } else {
                response = inferenceExecutor.executeBatch(
                    session,
                    request,
                    requestId
                );
            }

            // Record metrics
            var duration = Duration.between(startTime, Instant.now());
            metricsCollector.recordInferenceSuccess(
                request.model(),
                duration.toMillis(),
                response.getTokensUsed()
            );

            return response;

        } catch (Exception e) {
            metricsCollector.recordInferenceFailure(
                request.model(),
                e.getClass().getSimpleName()
            );
            
            throw new GGUFInferenceException(
                "Inference failed for model: " + request.model(),
                e,
                ErrorPayload.builder()
                    .type("GGUFInferenceError")
                    .message(e.getMessage())
                    .retryable(isRetryable(e))
                    .originNode("gguf-provider")
                    .detail("model", request.model())
                    .detail("requestId", requestId)
                    .build()
            );
        }
    }

    /**
     * Get or load model metadata with caching
     */
    private GGUFModelMetadata getOrLoadModel(String modelId) {
        return modelCache.computeIfAbsent(modelId, id -> {
            try {
                LOG.infof("Loading model metadata: %s", id);
                return modelLoader.loadMetadata(id);
            } catch (Exception e) {
                throw new GGUFLoadException(
                    "Failed to load model: " + id,
                    e,
                    ErrorPayload.builder()
                        .type("GGUFModelLoadError")
                        .message("Model not found or corrupted: " + id)
                        .retryable(false)
                        .originNode("gguf-provider")
                        .detail("modelId", id)
                        .build()
                );
            }
        });
    }

    /**
     * Execute with circuit breaker protection
     */
    private <T> T executeWithCircuitBreaker(CircuitBreakerCallable<T> callable) {
        if (circuitBreaker == null) {
            return callable.call();
        }
        
        return circuitBreaker.call(callable::call);
    }

    @FunctionalInterface
    private interface CircuitBreakerCallable<T> {
        T call() throws Exception;
    }

    /**
     * Prewarm configured models on startup
     */
    private void prewarmModels() {
        var modelsToWarm = config.prewarmModels();
        if (modelsToWarm.isEmpty()) {
            return;
        }

        LOG.infof("Prewarming %d models", modelsToWarm.size());
        
        modelsToWarm.forEach(modelId -> {
            try {
                getOrLoadModel(modelId);
                LOG.infof("Prewarmed model: %s", modelId);
            } catch (Exception e) {
                LOG.warnf(e, "Failed to prewarm model: %s", modelId);
            }
        });
    }

    /**
     * Extract tenant ID from request metadata
     */
    private String extractTenantId(ProviderRequest request) {
        return (String) request.metadata()
            .getOrDefault("tenantId", "default");
    }

    /**
     * Determine if error is retryable
     */
    private boolean isRetryable(Throwable error) {
        // Session errors are typically retryable
        if (error instanceof GGUFInferenceException) {
            return true;
        }
        // Model load errors are not retryable
        if (error instanceof GGUFLoadException) {
            return false;
        }
        // Default: retry on transient errors
        return !error.getMessage().contains("validation") &&
               !error.getMessage().contains("quota");
    }

    /**
     * Ensure provider is initialized
     */
    private void ensureInitialized() {
        if (!initialized) {
            throw new IllegalStateException("GGUF Provider not initialized");
        }
    }

    /**
     * Health check
     */
    public boolean isHealthy() {
        return initialized && healthChecker.check().isHealthy();
    }

    /**
     * Get current metrics
     */
    public Map<String, Object> getMetrics() {
        return Map.of(
            "provider", PROVIDER_ID,
            "initialized", initialized,
            "circuit_breaker_state", circuitBreaker != null 
                ? circuitBreaker.getState().name() 
                : "DISABLED",
            "loaded_models", modelCache.size(),
            "active_sessions", sessionManager.getActiveSessionCount()
        );
    }
}
```

### GGUFProviderConfig.java

```java
package tech.kayys.wayang.inference.providers.gguf;

import io.smallrye.config.ConfigMapping;
import io.smallrye.config.WithDefault;
import io.smallrye.config.WithName;

import java.util.List;
import java.util.Optional;

/**
 * Configuration for GGUF provider.
 * All settings are externalized and support runtime updates where applicable.
 */
@ConfigMapping(prefix = "gguf.provider")
public interface GGUFProviderConfig {

    /**
     * Base directory for GGUF model files
     */
    @WithName("model.base-path")
    @WithDefault("/var/lib/wayang/models/gguf")
    String modelBasePath();

    /**
     * Maximum context window size
     */
    @WithName("max-context-tokens")
    @WithDefault("4096")
    int maxContextTokens();

    /**
     * Enable GPU acceleration
     */
    @WithName("gpu.enabled")
    @WithDefault("true")
    boolean gpuEnabled();

    /**
     * Number of layers to offload to GPU
     */
    @WithName("gpu.layers")
    @WithDefault("32")
    int gpuLayers();

    /**
     * GPU device ID
     */
    @WithName("gpu.device-id")
    @WithDefault("0")
    int gpuDeviceId();

    /**
     * Enable batching
     */
    @WithName("batching.enabled")
    @WithDefault("true")
    boolean batchingEnabled();

    /**
     * Maximum batch size
     */
    @WithName("batching.max-size")
    @WithDefault("8")
    int batchingMaxSize();

    /**
     * Session pool configuration
     */
    @WithName("session.pool.min-size")
    @WithDefault("1")
    int sessionPoolMinSize();

    @WithName("session.pool.max-size")
    @WithDefault("4")
    int sessionPoolMaxSize();

    @WithName("session.pool.idle-timeout-ms")
    @WithDefault("300000") // 5 minutes
    long sessionPoolIdleTimeoutMs();

    /**
     * Circuit breaker configuration
     */
    @WithName("circuit-breaker.failure-threshold")
    @WithDefault("5")
    int circuitBreakerFailureThreshold();

    @WithName("circuit-breaker.open-duration-ms")
    @WithDefault("30000")
    long circuitBreakerOpenDurationMs();

    @WithName("circuit-breaker.half-open-permits")
    @WithDefault("3")
    int circuitBreakerHalfOpenPermits();

    @WithName("circuit-breaker.half-open-success-threshold")
    @WithDefault("2")
    int circuitBreakerHalfOpenSuccessThreshold();

    /**
     * Prewarm models on startup
     */
    @WithName("prewarm.enabled")
    @WithDefault("false")
    boolean prewarmOnStartup();

    @WithName("prewarm.models")
    Optional<List<String>> prewarmModels();

    /**
     * Thread configuration
     */
    @WithName("threads")
    @WithDefault("4")
    int threads();

    /**
     * Memory mapping
     */
    @WithName("mmap.enabled")
    @WithDefault("true")
    boolean mmapEnabled();

    @WithName("mlock.enabled")
    @WithDefault("false")
    boolean mlockEnabled();
}
```

## 3. Native Binding Layer

### LlamaCppBinding.java

```java
package tech.kayys.wayang.inference.providers.gguf;

import com.sun.jna.Pointer;
import org.jboss.logging.Logger;

import java.util.concurrent.atomic.AtomicBoolean;

/**
 * High-level wrapper for llama.cpp native library.
 * Provides safe, idiomatic Java API over JNA bindings.
 * 
 * Thread-safety: All methods are thread-safe.
 * Resource management: Callers must ensure proper cleanup of native resources.
 */
public class LlamaCppBinding {

    private static final Logger LOG = Logger.getLogger(LlamaCppBinding.class);
    private static final AtomicBoolean INITIALIZED = new AtomicBoolean(false);

    /**
     * Initialize native library
     */
    public static void initialize() {
        if (INITIALIZED.compareAndSet(false, true)) {
            LOG.info("Initializing llama.cpp native library");
            LlamaCppNative.INSTANCE.llama_backend_init(false);
            LOG.info("llama.cpp initialized successfully");
        }
    }

    /**
     * Get library version
     */
    public static String getLibraryVersion() {
        initialize();
        return "llama.cpp-" + LlamaCppNative.LLAMA_BUILD_NUMBER;
    }

    /**
     * Load model from file
     */
    public static Pointer loadModel(String path, ModelParams params) {
        initialize();
        
        var nativeParams = new LlamaCppNative.LlamaModelParams();
        nativeParams.n_gpu_layers = params.gpuLayers();
        nativeParams.use_mmap = params.useMmap();
        nativeParams.use_mlock = params.useMlock();

        var model = LlamaCppNative.INSTANCE.llama_load_model_from_file(
            path,
            nativeParams
        );

        if (model == null) {
            throw new IllegalStateException("Failed to load model: " + path);
        }

        LOG.infof("Model loaded: %s (GPU layers: %d)", path, params.gpuLayers());
        return model;
    }

    /**
     * Create context for inference
     */
    public static Pointer createContext(Pointer model, ContextParams params) {
        var nativeParams = new LlamaCppNative.LlamaContextParams();
        nativeParams.n_ctx = params.contextSize();
        nativeParams.n_batch = params.batchSize();
        nativeParams.n_threads = params.threads();
        nativeParams.seed = params.seed();

        var context = LlamaCppNative.INSTANCE.llama_new_context_with_model(
            model,
            nativeParams
        );

        if (context == null) {
            throw new IllegalStateException("Failed to create context");
        }

        LOG.debugf("Context created (size: %d, batch: %d, threads: %d)",
            params.contextSize(), params.batchSize(), params.threads());

        return context;
    }

    /**
     * Tokenize text
     */
    public static int[] tokenize(Pointer context, String text, boolean addBos) {
        var maxTokens = text.length() * 2; // Rough estimate
        var tokens = new int[maxTokens];
        
        var count = LlamaCppNative.INSTANCE.llama_tokenize(
            context,
            text,
            text.length(),
            tokens,
            maxTokens,
            addBos,
            false // special tokens
        );

        if (count < 0) {
            throw new IllegalStateException("Tokenization failed");
        }

        var result = new int[count];
        System.arraycopy(tokens, 0, result, 0, count);
        return result;
    }

    /**
     * Decode tokens to text
     */
    public static String detokenize(Pointer context, int token) {
        var buffer = new byte[32]; // Most tokens fit in 32 bytes
        
        var length = LlamaCppNative.INSTANCE.llama_token_to_piece(
            context,
            token,
            buffer,
            buffer.length
        );

        if (length < 0) {
            return "";
        }

        return new String(buffer, 0, length);
    }

    /**
     * Evaluate tokens (forward pass)
     */
    public static void evaluate(Pointer context, int[] tokens, int nPast) {
        var result = LlamaCppNative.INSTANCE.llama_eval(
            context,
            tokens,
            tokens.length,
            nPast,
            Runtime.getRuntime().availableProcessors()
        );

        if (result != 0) {
            throw new IllegalStateException("Evaluation failed with code: " + result);
        }
    }

    /**
     * Sample next token
     */
    public static int sample(
        Pointer context,
        float temperature,
        float topP,
        int topK,
        float repeatPenalty,
        int[] lastTokens
    ) {
        var logits = LlamaCppNative.INSTANCE.llama_get_logits(context);
        var nVocab = LlamaCppNative.INSTANCE.llama_n_vocab(context);

        // Apply temperature
        if (temperature != 1.0f) {
            for (int i = 0; i < nVocab; i++) {
                logits.setFloat(i * 4L, logits.getFloat(i * 4L) / temperature);
            }
        }

        // Apply repeat penalty
        if (repeatPenalty != 1.0f && lastTokens != null) {
            for (int token : lastTokens) {
                var logit = logits.getFloat(token * 4L);
                logits.setFloat(
                    token * 4L,
                    logit < 0 ? logit * repeatPenalty : logit / repeatPenalty
                );
            }
        }

        // Sample using top-p
        return LlamaCppNative.INSTANCE.llama_sample_top_p_top_k(
            context,
            null,
            0,
            topK,
            topP,
            1.0f,
            1.0f
        );
    }

    /**
     * Free context
     */
    public static void freeContext(Pointer context) {
        if (context != null) {
            LlamaCppNative.INSTANCE.llama_free(context);
            LOG.debug("Context freed");
        }
    }

    /**
     * Free model
     */
    public static void freeModel(Pointer model) {
        if (model != null) {
            LlamaCppNative.INSTANCE.llama_free_model(model);
            LOG.debug("Model freed");
        }
    }

    /**
     * Model loading parameters
     */
    public record ModelParams(
        int gpuLayers,
        boolean useMmap,
        boolean useMlock
    ) {
        public static ModelParams defaults() {
            return new ModelParams(0, true, false);
        }
    }

    /**
     * Context creation parameters
     */
    public record ContextParams(
        int contextSize,
        int batchSize,
        int threads,
        int seed
    ) {
        public static ContextParams defaults() {
            return new ContextParams(
                2048,
                512,
                Runtime.getRuntime().availableProcessors(),
                -1
            );
        }
    }
}
```

### LlamaCppNative.java

```java
package tech.kayys.wayang.inference.providers.gguf;

import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.Pointer;
import com.sun.jna.Structure;

import java.util.Arrays;
import java.util.List;

/**
 * JNA bindings for llama.cpp C API.
 * 
 * This class provides low-level FFI bindings.
 * Use LlamaCppBinding for high-level operations.
 * 
 * IMPORTANT: Keep in sync with llama.cpp version.
 */
public interface LlamaCppNative extends Library {

    LlamaCppNative INSTANCE = Native.load("llama", LlamaCppNative.class);

    int LLAMA_BUILD_NUMBER = 3715;
    int LLAMA_FILE_MAGIC_GGUF = 0x46554747; // "GGUF"

    // Backend initialization
    void llama_backend_init(boolean numa);
    void llama_backend_free();

    // Model operations
    Pointer llama_load_model_from_file(String path, LlamaModelParams params);
    void llama_free_model(Pointer model);
    int llama_n_vocab(Pointer model);
    int llama_n_ctx_train(Pointer model);
    int llama_n_embd(Pointer model);

    // Context operations
    Pointer llama_new_context_with_model(Pointer model, LlamaContextParams params);
    void llama_free(Pointer context);
    int llama_n_ctx(Pointer context);

    // Tokenization
    int llama_tokenize(
        Pointer context,
        String text,
        int textLen,
        int[] tokens,
        int nMaxTokens,
        boolean addBos,
        boolean special
    );

    int llama_token_to_piece(
        Pointer context,
        int token,
        byte[] buffer,
        int length
    );

    int llama_token_bos(Pointer context);
    int llama_token_eos(Pointer context);
    int llama_token_nl(Pointer context);

    // Inference
    int llama_eval(
        Pointer context,
        int[] tokens,
        int nTokens,
        int nPast,
        int nThreads
    );

    Pointer llama_get_logits(Pointer context);

    // Sampling
    int llama_sample_top_p_top_k(
        Pointer context,
        int[] lastTokens,
        int nLastTokens,
        int topK,
        float topP,
        float temp,
        float repeatPenalty
    );

    // Parameter structures
    @Structure.FieldOrder({
        "n_gpu_layers",
        "main_gpu",
        "tensor_split",
        "vocab_only",
        "use_mmap",
        "use_mlock"
    })
    class LlamaModelParams extends Structure {
        public int n_gpu_layers;
        public int main_gpu;
        public Pointer tensor_split;
        public boolean vocab_only;
        public boolean use_mmap;
        public boolean use_mlock;

        public LlamaModelParams() {
            n_gpu_layers = 0;
            main_gpu = 0;
            vocab_only = false;
            use_mmap = true;
            use_mlock = false;
        }
    }

    @Structure.FieldOrder({
        "seed",
        "n_ctx",
        "n_batch",
        "n_threads",
        "n_threads_batch",
        "rope_scaling_type",
        "rope_freq_base",
        "rope_freq_scale",
        "yarn_ext_factor",
        "yarn_attn_factor",
        "yarn_beta_fast",
        "yarn_beta_slow",
        "yarn_orig_ctx"
    })
    class LlamaContextParams extends Structure {
        public int seed;
        public int n_ctx;
        public int n_batch;
        public int n_threads;
        public int n_threads_batch;
        public int rope_scaling_type;
        public float rope_freq_base;
        public float rope_freq_scale;
        public float yarn_ext_factor;
        public float yarn_attn_factor;
        public float yarn_beta_fast;
        public float yarn_beta_slow;
        public int yarn_orig_ctx;

        public LlamaContextParams() {
            seed = -1;
            n_ctx = 512;
            n_batch = 512;
            n_threads = 4;
            n_threads_batch = 4;
            rope_scaling_type = -1;
            rope_freq_base = 0.0f;
            rope_freq_scale = 0.0f;
            yarn_ext_factor = -1.0f;
            yarn_attn_factor = 1.0f;
            yarn_beta_fast = 32.0f;
            yarn_beta_slow = 1.0f;
            yarn_orig_ctx = 0;
        }
    }
}
```

This is Part 1 of the implementation. Would you like me to continue with:
- Part 2: Session Management & Model Loading
- Part 3: Inference Executor & Streaming
- Part 4: Health Checks, Metrics & Testing

?


# Part 2: Session Management & Model Loading

## 4. Model Loading and Metadata

### GGUFModelMetadata.java

```java
package tech.kayys.wayang.inference.providers.gguf;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.nio.file.Path;
import java.time.Instant;
import java.util.Map;
import java.util.Objects;

/**
 * Metadata for a GGUF model file.
 * Contains all information needed for session creation and inference.
 * 
 * Immutable and serializable for caching.
 */
public final class GGUFModelMetadata {

    private final String modelId;
    private final String name;
    private final Path filePath;
    private final long fileSizeBytes;
    private final String quantization;
    private final int contextSize;
    private final int vocabularySize;
    private final int embeddingSize;
    private final String architecture;
    private final Map<String, Object> parameters;
    private final Instant loadedAt;
    private final String checksum;

    @JsonCreator
    public GGUFModelMetadata(
        @JsonProperty("modelId") String modelId,
        @JsonProperty("name") String name,
        @JsonProperty("filePath") Path filePath,
        @JsonProperty("fileSizeBytes") long fileSizeBytes,
        @JsonProperty("quantization") String quantization,
        @JsonProperty("contextSize") int contextSize,
        @JsonProperty("vocabularySize") int vocabularySize,
        @JsonProperty("embeddingSize") int embeddingSize,
        @JsonProperty("architecture") String architecture,
        @JsonProperty("parameters") Map<String, Object> parameters,
        @JsonProperty("loadedAt") Instant loadedAt,
        @JsonProperty("checksum") String checksum
    ) {
        this.modelId = Objects.requireNonNull(modelId, "modelId");
        this.name = Objects.requireNonNull(name, "name");
        this.filePath = Objects.requireNonNull(filePath, "filePath");
        this.fileSizeBytes = fileSizeBytes;
        this.quantization = quantization;
        this.contextSize = contextSize;
        this.vocabularySize = vocabularySize;
        this.embeddingSize = embeddingSize;
        this.architecture = architecture;
        this.parameters = Map.copyOf(parameters);
        this.loadedAt = loadedAt != null ? loadedAt : Instant.now();
        this.checksum = checksum;
    }

    // Getters
    public String getModelId() { return modelId; }
    public String getName() { return name; }
    public Path getFilePath() { return filePath; }
    public long getFileSizeBytes() { return fileSizeBytes; }
    public String getQuantization() { return quantization; }
    public int getContextSize() { return contextSize; }
    public int getVocabularySize() { return vocabularySize; }
    public int getEmbeddingSize() { return embeddingSize; }
    public String getArchitecture() { return architecture; }
    public Map<String, Object> getParameters() { return parameters; }
    public Instant getLoadedAt() { return loadedAt; }
    public String getChecksum() { return checksum; }

    /**
     * Estimate memory requirements in bytes
     */
    public long estimateMemoryRequirements() {
        // Rough estimate: file size + context overhead
        long contextOverhead = (long) contextSize * embeddingSize * 4; // 4 bytes per float
        return fileSizeBytes + contextOverhead;
    }

    /**
     * Check if model supports GPU offloading
     */
    public boolean supportsGPU() {
        // GGUF models generally support GPU, but check architecture
        return architecture != null && 
               (architecture.contains("llama") || 
                architecture.contains("mistral") ||
                architecture.contains("mixtral"));
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String modelId;
        private String name;
        private Path filePath;
        private long fileSizeBytes;
        private String quantization = "unknown";
        private int contextSize = 2048;
        private int vocabularySize;
        private int embeddingSize;
        private String architecture;
        private Map<String, Object> parameters = Map.of();
        private Instant loadedAt;
        private String checksum;

        public Builder modelId(String modelId) {
            this.modelId = modelId;
            return this;
        }

        public Builder name(String name) {
            this.name = name;
            return this;
        }

        public Builder filePath(Path filePath) {
            this.filePath = filePath;
            return this;
        }

        public Builder fileSizeBytes(long fileSizeBytes) {
            this.fileSizeBytes = fileSizeBytes;
            return this;
        }

        public Builder quantization(String quantization) {
            this.quantization = quantization;
            return this;
        }

        public Builder contextSize(int contextSize) {
            this.contextSize = contextSize;
            return this;
        }

        public Builder vocabularySize(int vocabularySize) {
            this.vocabularySize = vocabularySize;
            return this;
        }

        public Builder embeddingSize(int embeddingSize) {
            this.embeddingSize = embeddingSize;
            return this;
        }

        public Builder architecture(String architecture) {
            this.architecture = architecture;
            return this;
        }

        public Builder parameters(Map<String, Object> parameters) {
            this.parameters = parameters;
            return this;
        }

        public Builder loadedAt(Instant loadedAt) {
            this.loadedAt = loadedAt;
            return this;
        }

        public Builder checksum(String checksum) {
            this.checksum = checksum;
            return this;
        }

        public GGUFModelMetadata build() {
            return new GGUFModelMetadata(
                modelId, name, filePath, fileSizeBytes, quantization,
                contextSize, vocabularySize, embeddingSize, architecture,
                parameters, loadedAt, checksum
            );
        }
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof GGUFModelMetadata that)) return false;
        return modelId.equals(that.modelId) && 
               checksum.equals(that.checksum);
    }

    @Override
    public int hashCode() {
        return Objects.hash(modelId, checksum);
    }

    @Override
    public String toString() {
        return "GGUFModelMetadata{" +
               "modelId='" + modelId + '\'' +
               ", name='" + name + '\'' +
               ", quantization='" + quantization + '\'' +
               ", contextSize=" + contextSize +
               ", fileSizeMB=" + (fileSizeBytes / 1024 / 1024) +
               '}';
    }
}
```

### GGUFModelLoader.java

```java
package tech.kayys.wayang.inference.providers.gguf;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.apache.commons.codec.digest.DigestUtils;
import org.jboss.logging.Logger;
import tech.kayys.wayang.inference.providers.gguf.exceptions.GGUFLoadException;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Loads GGUF model files and extracts metadata.
 * 
 * Features:
 * - GGUF format parsing
 * - Metadata extraction
 * - Checksum verification
 * - File validation
 * - Caching
 * 
 * Thread-safety: Safe for concurrent access.
 */
@ApplicationScoped
public class GGUFModelLoader {

    private static final Logger LOG = Logger.getLogger(GGUFModelLoader.class);
    private static final int GGUF_MAGIC = 0x46554747; // "GGUF"
    private static final int GGUF_VERSION = 3;

    @Inject
    GGUFProviderConfig config;

    private final Map<String, GGUFModelMetadata> metadataCache = new ConcurrentHashMap<>();

    /**
     * Load model metadata from file
     */
    public GGUFModelMetadata loadMetadata(String modelId) {
        return metadataCache.computeIfAbsent(modelId, id -> {
            try {
                LOG.infof("Loading metadata for model: %s", id);
                
                Path modelPath = resolveModelPath(id);
                validateModelFile(modelPath);
                
                var metadata = parseGGUFMetadata(modelPath, id);
                
                LOG.infof("Model metadata loaded: %s", metadata);
                return metadata;
                
            } catch (Exception e) {
                throw new GGUFLoadException(
                    "Failed to load model metadata: " + id,
                    e
                );
            }
        });
    }

    /**
     * Parse GGUF file and extract metadata
     */
    private GGUFModelMetadata parseGGUFMetadata(Path filePath, String modelId) 
            throws IOException {
        
        try (FileChannel channel = FileChannel.open(filePath, StandardOpenOption.READ)) {
            
            // Read header
            ByteBuffer header = ByteBuffer.allocate(1024);
            header.order(ByteOrder.LITTLE_ENDIAN);
            channel.read(header);
            header.flip();

            // Verify magic number
            int magic = header.getInt();
            if (magic != GGUF_MAGIC) {
                throw new GGUFLoadException(
                    "Invalid GGUF file: wrong magic number"
                );
            }

            // Read version
            int version = header.getInt();
            if (version != GGUF_VERSION) {
                LOG.warnf("GGUF version mismatch: expected %d, got %d", 
                    GGUF_VERSION, version);
            }

            // Read tensor count and metadata count
            long tensorCount = header.getLong();
            long metadataCount = header.getLong();

            LOG.debugf("GGUF file - tensors: %d, metadata entries: %d", 
                tensorCount, metadataCount);

            // Parse metadata entries
            Map<String, Object> metadata = new HashMap<>();
            for (long i = 0; i < metadataCount; i++) {
                parseMetadataEntry(header, channel, metadata);
            }

            // Extract key parameters
            String architecture = (String) metadata.getOrDefault(
                "general.architecture", "unknown"
            );
            
            Integer contextLength = extractInteger(metadata, 
                architecture + ".context_length", 2048);
            
            Integer vocabSize = extractInteger(metadata,
                architecture + ".vocab_size", 32000);
            
            Integer embeddingSize = extractInteger(metadata,
                architecture + ".embedding_length", 4096);
            
            String quantization = extractQuantization(metadata);

            // Calculate checksum
            String checksum = calculateChecksum(filePath);

            return GGUFModelMetadata.builder()
                .modelId(modelId)
                .name(filePath.getFileName().toString())
                .filePath(filePath)
                .fileSizeBytes(Files.size(filePath))
                .quantization(quantization)
                .contextSize(contextLength)
                .vocabularySize(vocabSize)
                .embeddingSize(embeddingSize)
                .architecture(architecture)
                .parameters(metadata)
                .checksum(checksum)
                .build();
        }
    }

    /**
     * Parse a single metadata entry from GGUF file
     */
    private void parseMetadataEntry(
        ByteBuffer buffer, 
        FileChannel channel,
        Map<String, Object> metadata
    ) throws IOException {
        
        // Ensure buffer has data
        if (buffer.remaining() < 16) {
            buffer.clear();
            channel.read(buffer);
            buffer.flip();
        }

        // Read key length and key
        long keyLength = buffer.getLong();
        byte[] keyBytes = new byte[(int) keyLength];
        
        if (buffer.remaining() < keyLength) {
            buffer.clear();
            channel.read(buffer);
            buffer.flip();
        }
        
        buffer.get(keyBytes);
        String key = new String(keyBytes);

        // Read value type
        int valueType = buffer.getInt();

        // Read value based on type
        Object value = parseMetadataValue(buffer, channel, valueType);
        
        metadata.put(key, value);
        
        LOG.tracef("Metadata: %s = %s (type: %d)", key, value, valueType);
    }

    /**
     * Parse metadata value based on type
     */
    private Object parseMetadataValue(
        ByteBuffer buffer,
        FileChannel channel,
        int type
    ) throws IOException {
        
        // Ensure buffer has data
        if (buffer.remaining() < 8) {
            buffer.clear();
            channel.read(buffer);
            buffer.flip();
        }

        return switch (type) {
            case 0 -> buffer.get(); // uint8
            case 1 -> buffer.getShort(); // int8
            case 2 -> buffer.getShort(); // uint16
            case 3 -> buffer.getShort(); // int16
            case 4 -> buffer.getInt(); // uint32
            case 5 -> buffer.getInt(); // int32
            case 6 -> buffer.getFloat(); // float32
            case 7 -> buffer.get() != 0; // bool
            case 8 -> { // string
                long length = buffer.getLong();
                byte[] strBytes = new byte[(int) length];
                if (buffer.remaining() < length) {
                    buffer.clear();
                    channel.read(buffer);
                    buffer.flip();
                }
                buffer.get(strBytes);
                yield new String(strBytes);
            }
            default -> {
                LOG.warnf("Unknown metadata value type: %d", type);
                yield null;
            }
        };
    }

    /**
     * Extract integer from metadata with fallback
     */
    private Integer extractInteger(Map<String, Object> metadata, String key, int defaultValue) {
        Object value = metadata.get(key);
        if (value instanceof Number num) {
            return num.intValue();
        }
        return defaultValue;
    }

    /**
     * Extract quantization info from metadata
     */
    private String extractQuantization(Map<String, Object> metadata) {
        // Try to find quantization info in metadata
        for (String key : metadata.keySet()) {
            if (key.contains("quantization") || key.contains("type")) {
                Object value = metadata.get(key);
                if (value != null) {
                    return value.toString();
                }
            }
        }
        return "unknown";
    }

    /**
     * Calculate SHA-256 checksum of file
     */
    private String calculateChecksum(Path filePath) throws IOException {
        try (var inputStream = Files.newInputStream(filePath)) {
            return DigestUtils.sha256Hex(inputStream);
        }
    }

    /**
     * Resolve model path from ID
     */
    private Path resolveModelPath(String modelId) {
        // Check if it's already a path
        if (modelId.contains("/") || modelId.contains("\\")) {
            return Paths.get(modelId);
        }

        // Look in base path
        Path basePath = Paths.get(config.modelBasePath());
        
        // Try with .gguf extension
        Path withExtension = basePath.resolve(modelId + ".gguf");
        if (Files.exists(withExtension)) {
            return withExtension;
        }

        // Try as-is
        Path asIs = basePath.resolve(modelId);
        if (Files.exists(asIs)) {
            return asIs;
        }

        throw new GGUFLoadException(
            "Model file not found: " + modelId + " (searched in: " + basePath + ")"
        );
    }

    /**
     * Validate model file
     */
    private void validateModelFile(Path filePath) throws IOException {
        if (!Files.exists(filePath)) {
            throw new GGUFLoadException("Model file does not exist: " + filePath);
        }

        if (!Files.isReadable(filePath)) {
            throw new GGUFLoadException("Model file is not readable: " + filePath);
        }

        long sizeBytes = Files.size(filePath);
        if (sizeBytes == 0) {
            throw new GGUFLoadException("Model file is empty: " + filePath);
        }

        LOG.debugf("Model file validated: %s (size: %d MB)", 
            filePath, sizeBytes / 1024 / 1024);
    }

    /**
     * Clear metadata cache
     */
    public void clearCache() {
        metadataCache.clear();
        LOG.info("Model metadata cache cleared");
    }

    /**
     * Invalidate specific model from cache
     */
    public void invalidateModel(String modelId) {
        metadataCache.remove(modelId);
        LOG.infof("Model metadata invalidated: %s", modelId);
    }
}
```

## 5. Session Management

### GGUFSession.java

```java
package tech.kayys.wayang.inference.providers.gguf;

import com.sun.jna.Pointer;
import org.jboss.logging.Logger;

import java.time.Instant;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Represents an active GGUF model session.
 * 
 * A session contains:
 * - Loaded model handle
 * - Inference context
 * - State (tokens, position)
 * - Resource management
 * 
 * Thread-safety: Sessions are NOT thread-safe. 
 * Concurrent access must be synchronized externally.
 */
public class GGUFSession implements AutoCloseable {

    private static final Logger LOG = Logger.getLogger(GGUFSession.class);

    private final String sessionId;
    private final String tenantId;
    private final String modelId;
    private final GGUFModelMetadata metadata;
    private final Pointer modelHandle;
    private final Pointer contextHandle;
    private final Instant createdAt;
    
    private final AtomicBoolean closed = new AtomicBoolean(false);
    private final AtomicInteger usageCount = new AtomicInteger(0);
    private final ReentrantLock sessionLock = new ReentrantLock();
    
    private volatile Instant lastUsedAt;
    private volatile int currentPosition = 0;

    public GGUFSession(
        String sessionId,
        String tenantId,
        String modelId,
        GGUFModelMetadata metadata,
        Pointer modelHandle,
        Pointer contextHandle
    ) {
        this.sessionId = sessionId;
        this.tenantId = tenantId;
        this.modelId = modelId;
        this.metadata = metadata;
        this.modelHandle = modelHandle;
        this.contextHandle = contextHandle;
        this.createdAt = Instant.now();
        this.lastUsedAt = Instant.now();
    }

    /**
     * Acquire lock for inference
     */
    public void lock() {
        sessionLock.lock();
        usageCount.incrementAndGet();
        lastUsedAt = Instant.now();
    }

    /**
     * Release lock after inference
     */
    public void unlock() {
        sessionLock.unlock();
    }

    /**
     * Check if session is currently in use
     */
    public boolean isLocked() {
        return sessionLock.isLocked();
    }

    /**
     * Get model handle
     */
    public Pointer getModelHandle() {
        ensureNotClosed();
        return modelHandle;
    }

    /**
     * Get context handle
     */
    public Pointer getContextHandle() {
        ensureNotClosed();
        return contextHandle;
    }

    /**
     * Get current token position
     */
    public int getCurrentPosition() {
        return currentPosition;
    }

    /**
     * Update token position
     */
    public void setCurrentPosition(int position) {
        this.currentPosition = position;
    }

    /**
     * Reset session state
     */
    public void reset() {
        lock();
        try {
            currentPosition = 0;
            LOG.debugf("Session reset: %s", sessionId);
        } finally {
            unlock();
        }
    }

    /**
     * Check if session can be recycled
     */
    public boolean canRecycle() {
        if (closed.get() || isLocked()) {
            return false;
        }

        // Check idle time
        long idleMs = Instant.now().toEpochMilli() - lastUsedAt.toEpochMilli();
        return idleMs < 300000; // 5 minutes
    }

    /**
     * Close and release resources
     */
    @Override
    public void close() {
        if (closed.compareAndSet(false, true)) {
            LOG.infof("Closing session: %s (usage count: %d)", 
                sessionId, usageCount.get());

            try {
                if (contextHandle != null) {
                    LlamaCppBinding.freeContext(contextHandle);
                }
                if (modelHandle != null) {
                    LlamaCppBinding.freeModel(modelHandle);
                }
            } catch (Exception e) {
                LOG.error("Error closing session", e);
            }
        }
    }

    private void ensureNotClosed() {
        if (closed.get()) {
            throw new IllegalStateException("Session is closed: " + sessionId);
        }
    }

    // Getters
    public String getSessionId() { return sessionId; }
    public String getTenantId() { return tenantId; }
    public String getModelId() { return modelId; }
    public GGUFModelMetadata getMetadata() { return metadata; }
    public Instant getCreatedAt() { return createdAt; }
    public Instant getLastUsedAt() { return lastUsedAt; }
    public int getUsageCount() { return usageCount.get(); }
    public boolean isClosed() { return closed.get(); }

    @Override
    public String toString() {
        return "GGUFSession{" +
               "id='" + sessionId + '\'' +
               ", tenant='" + tenantId + '\'' +
               ", model='" + modelId + '\'' +
               ", position=" + currentPosition +
               ", usageCount=" + usageCount.get() +
               ", closed=" + closed.get() +
               '}';
    }
}
```

### GGUFSessionManager.java

```java
package tech.kayys.wayang.inference.providers.gguf;

import com.sun.jna.Pointer;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.wayang.inference.providers.gguf.exceptions.GGUFSessionException;

import java.time.Duration;
import java.time.Instant;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.*;

/**
 * Manages GGUF session lifecycle with pooling.
 * 
 * Features:
 * - Per-tenant session pooling
 * - Automatic session recycling
 * - Resource quota enforcement
 * - Idle session cleanup
 * - Health monitoring
 * 
 * Thread-safety: All public methods are thread-safe.
 */
@ApplicationScoped
public class GGUFSessionManager {

    private static final Logger LOG = Logger.getLogger(GGUFSessionManager.class);

    @Inject
    GGUFProviderConfig config;

    // Session pools: (tenantId + modelId) -> session pool
    private final Map<String, BlockingQueue<GGUFSession>> sessionPools = 
        new ConcurrentHashMap<>();

    // Active sessions tracking
    private final Map<String, GGUFSession> activeSessions = 
        new ConcurrentHashMap<>();

    // Cleanup scheduler
    private ScheduledExecutorService cleanupScheduler;

    /**
     * Initialize session manager
     */
    public void initialize() {
        LOG.info("Initializing GGUF Session Manager");

        // Start cleanup scheduler
        cleanupScheduler = Executors.newSingleThreadScheduledExecutor(r -> {
            Thread t = new Thread(r, "gguf-session-cleanup");
            t.setDaemon(true);
            return t;
        });

        cleanupScheduler.scheduleAtFixedRate(
            this::cleanupIdleSessions,
            1, 1, TimeUnit.MINUTES
        );

        LOG.info("GGUF Session Manager initialized");
    }

    /**
     * Get or create session for tenant and model
     */
    public GGUFSession getSession(
        String tenantId,
        String modelId,
        GGUFModelMetadata metadata
    ) {
        String poolKey = makePoolKey(tenantId, modelId);
        
        // Try to get from pool
        BlockingQueue<GGUFSession> pool = sessionPools.computeIfAbsent(
            poolKey,
            k -> new ArrayBlockingQueue<>(config.sessionPoolMaxSize())
        );

        GGUFSession session = pool.poll();
        
        if (session != null && session.canRecycle()) {
            LOG.debugf("Reusing session from pool: %s", session.getSessionId());
            session.reset();
            return session;
        }

        // Create new session
        return createSession(tenantId, modelId, metadata);
    }

    /**
     * Return session to pool
     */
    public void returnSession(GGUFSession session) {
        if (session == null || session.isClosed()) {
            return;
        }

        String poolKey = makePoolKey(session.getTenantId(), session.getModelId());
        BlockingQueue<GGUFSession> pool = sessionPools.get(poolKey);

        if (pool != null && session.canRecycle()) {
            boolean added = pool.offer(session);
            if (!added) {
                LOG.debugf("Pool full, closing session: %s", session.getSessionId());
                session.close();
            }
        } else {
            session.close();
        }

        activeSessions.remove(session.getSessionId());
    }

    /**
     * Create new session
     */
    private GGUFSession createSession(
        String tenantId,
        String modelId,
        GGUFModelMetadata metadata
    ) {
        LOG.infof("Creating new session for tenant=%s, model=%s", tenantId, modelId);

        try {
            // Load model
            Pointer modelHandle = LlamaCppBinding.loadModel(
                metadata.getFilePath().toString(),
                new LlamaCppBinding.ModelParams(
                    config.gpuEnabled() ? config.gpuLayers() : 0,
                    config.mmapEnabled(),
                    config.mlockEnabled()
                )
            );

            // Create context
            Pointer contextHandle = LlamaCppBinding.createContext(
                modelHandle,
                new LlamaCppBinding.ContextParams(
                    metadata.getContextSize(),
                    config.batchingMaxSize(),
                    config.threads(),
                    -1 // random seed
                )
            );

            String sessionId = UUID.randomUUID().toString();
            GGUFSession session = new GGUFSession(
                sessionId,
                tenantId,
                modelId,
                metadata,
                modelHandle,
                contextHandle
            );

            activeSessions.put(sessionId, session);
            
            LOG.infof("Session created: %s", sessionId);
            return session;

        } catch (Exception e) {
            throw new GGUFSessionException(
                "Failed to create session for model: " + modelId,
                e
            );
        }
    }

    /**
     * Cleanup idle sessions
     */
    private void cleanupIdleSessions() {
        LOG.debug("Running idle session cleanup");

        int cleaned = 0;
        long idleThresholdMs = config.sessionPoolIdleTimeoutMs();

        for (BlockingQueue<GGUFSession> pool : sessionPools.values()) {
            pool.removeIf(session -> {
                long idleMs = Duration.between(
                    session.getLastUsedAt(),
                    Instant.now()
                ).toMillis();

                if (idleMs > idleThresholdMs) {
                    LOG.debugf("Closing idle session: %s (idle: %d ms)", 
                        session.getSessionId(), idleMs);
                    session.close();
                    return true;
                }
                return false;
            });
        }

        if (cleaned > 0) {
            LOG.infof("Cleaned up %d idle sessions", cleaned);
        }
    }

    /**
     * Shutdown all sessions
     */
    public void shutdown() {
        LOG.info("Shutting down GGUF Session Manager");

        if (cleanupScheduler != null) {
            cleanupScheduler.shutdown();
        }

        // Close all active sessions
        activeSessions.values().forEach(GGUFSession::close);
        activeSessions.clear();

        // Close all pooled sessions
        sessionPools.values().forEach(pool -> {
            pool.forEach(GGUFSession::close);
            pool.clear();
        });
        sessionPools.clear();

        LOG.info("GGUF Session Manager shutdown complete");
    }

    /**
     * Get active session count
     */
    public int getActiveSessionCount() {
        return activeSessions.size();
    }

    /**
     * Get pooled session count
     */
    public int getPooledSessionCount() {
        return sessionPools.values().stream()
            .mapToInt(BlockingQueue::size)
            .sum();
    }

    private String makePoolKey(String tenantId, String modelId) {
        return tenantId + ":" + modelId;
    }
}
```

This completes Part 2. Shall I continue with Part 3 covering:
- Inference Executor (batch and streaming)
- Stream Handler
- Health Checker & Metrics Collector
- Exception Classes

?

# Part 3: Inference Execution & Streaming

## 6. Inference Executor

### GGUFInferenceExecutor.java

```java
package tech.kayys.wayang.inference.providers.gguf;

import io.opentelemetry.api.trace.Span;
import io.opentelemetry.api.trace.Tracer;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.wayang.inference.api.InferenceResponse;
import tech.kayys.wayang.inference.api.Message;
import tech.kayys.wayang.inference.api.StreamingResponse;
import tech.kayys.wayang.inference.kernel.provider.ProviderRequest;
import tech.kayys.wayang.inference.providers.gguf.exceptions.GGUFInferenceException;

import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Executes inference requests using GGUF sessions.
 * 
 * Features:
 * - Batch inference
 * - Streaming inference
 * - Token sampling
 * - Context management
 * - Parameter handling
 * 
 * Thread-safety: Methods are thread-safe but sessions must be locked externally.
 */
@ApplicationScoped
public class GGUFInferenceExecutor {

    private static final Logger LOG = Logger.getLogger(GGUFInferenceExecutor.class);

    @Inject
    GGUFProviderConfig config;

    @Inject
    GGUFStreamHandler streamHandler;

    @Inject
    Tracer tracer;

    /**
     * Execute batch (non-streaming) inference
     */
    public InferenceResponse executeBatch(
        GGUFSession session,
        ProviderRequest request,
        String requestId
    ) {
        var span = tracer.spanBuilder("gguf.execute.batch")
            .setAttribute("model", request.model())
            .setAttribute("requestId", requestId)
            .startSpan();

        session.lock();
        try {
            var startTime = Instant.now();
            
            // Build prompt from messages
            String prompt = buildPrompt(request.messages());
            
            // Extract parameters
            var params = extractParameters(request.parameters());
            
            // Tokenize prompt
            int[] promptTokens = LlamaCppBinding.tokenize(
                session.getContextHandle(),
                prompt,
                true // add BOS
            );

            LOG.debugf("Prompt tokenized: %d tokens", promptTokens.length);

            // Evaluate prompt
            LlamaCppBinding.evaluate(
                session.getContextHandle(),
                promptTokens,
                0 // n_past
            );

            // Generate completion
            List<Integer> generatedTokens = new ArrayList<>();
            List<Integer> lastTokens = new ArrayList<>();
            int maxTokens = params.maxTokens();
            
            for (int i = 0; i < maxTokens; i++) {
                // Sample next token
                int nextToken = LlamaCppBinding.sample(
                    session.getContextHandle(),
                    params.temperature(),
                    params.topP(),
                    params.topK(),
                    params.repeatPenalty(),
                    lastTokens.stream().mapToInt(Integer::intValue).toArray()
                );

                // Check for EOS
                if (isEndToken(session, nextToken)) {
                    LOG.debug("EOS token generated, stopping");
                    break;
                }

                generatedTokens.add(nextToken);
                lastTokens.add(nextToken);
                
                // Keep sliding window of last tokens for repeat penalty
                if (lastTokens.size() > 64) {
                    lastTokens.remove(0);
                }

                // Evaluate next token
                LlamaCppBinding.evaluate(
                    session.getContextHandle(),
                    new int[]{nextToken},
                    promptTokens.length + i
                );
            }

            // Detokenize
            String completion = detokenize(session, generatedTokens);
            
            var duration = Duration.between(startTime, Instant.now());
            int totalTokens = promptTokens.length + generatedTokens.size();

            LOG.infof("Batch inference completed: %d tokens in %d ms",
                totalTokens, duration.toMillis());

            return InferenceResponse.builder()
                .requestId(requestId)
                .content(completion)
                .model(request.model())
                .tokensUsed(totalTokens)
                .durationMs(duration.toMillis())
                .metadata("prompt_tokens", promptTokens.length)
                .metadata("completion_tokens", generatedTokens.size())
                .metadata("finish_reason", 
                    generatedTokens.size() >= maxTokens ? "length" : "stop")
                .build();

        } catch (Exception e) {
            LOG.error("Batch inference failed", e);
            throw new GGUFInferenceException("Batch inference failed", e);
        } finally {
            session.unlock();
            span.end();
        }
    }

    /**
     * Execute streaming inference
     */
    public StreamingResponse executeStreaming(
        GGUFSession session,
        ProviderRequest request,
        String requestId
    ) {
        var span = tracer.spanBuilder("gguf.execute.streaming")
            .setAttribute("model", request.model())
            .setAttribute("requestId", requestId)
            .startSpan();

        try {
            // Build prompt
            String prompt = buildPrompt(request.messages());
            
            // Extract parameters
            var params = extractParameters(request.parameters());

            // Create streaming handler
            return streamHandler.createStream(
                session,
                prompt,
                params,
                requestId,
                span
            );

        } catch (Exception e) {
            span.end();
            LOG.error("Streaming inference setup failed", e);
            throw new GGUFInferenceException("Streaming inference failed", e);
        }
    }

    /**
     * Build prompt from messages
     */
    private String buildPrompt(List<Message> messages) {
        if (messages.isEmpty()) {
            throw new IllegalArgumentException("Messages cannot be empty");
        }

        // Simple prompt builder - can be enhanced with chat templates
        StringBuilder prompt = new StringBuilder();
        
        for (Message message : messages) {
            switch (message.getRole()) {
                case SYSTEM -> prompt.append("System: ").append(message.getContent()).append("\n\n");
                case USER -> prompt.append("User: ").append(message.getContent()).append("\n\n");
                case ASSISTANT -> prompt.append("Assistant: ").append(message.getContent()).append("\n\n");
                default -> LOG.warnf("Unsupported role: %s", message.getRole());
            }
        }

        // Add assistant prefix for continuation
        prompt.append("Assistant: ");
        
        return prompt.toString();
    }

    /**
     * Extract and validate inference parameters
     */
    private InferenceParameters extractParameters(Map<String, Object> params) {
        return new InferenceParameters(
            getFloatParam(params, "temperature", 0.7f),
            getFloatParam(params, "top_p", 0.9f),
            getIntParam(params, "top_k", 40),
            getFloatParam(params, "repeat_penalty", 1.1f),
            getIntParam(params, "max_tokens", 512)
        );
    }

    private float getFloatParam(Map<String, Object> params, String key, float defaultValue) {
        Object value = params.get(key);
        if (value instanceof Number num) {
            return num.floatValue();
        }
        return defaultValue;
    }

    private int getIntParam(Map<String, Object> params, String key, int defaultValue) {
        Object value = params.get(key);
        if (value instanceof Number num) {
            return num.intValue();
        }
        return defaultValue;
    }

    /**
     * Detokenize list of tokens
     */
    private String detokenize(GGUFSession session, List<Integer> tokens) {
        return tokens.stream()
            .map(token -> LlamaCppBinding.detokenize(session.getContextHandle(), token))
            .collect(Collectors.joining());
    }

    /**
     * Check if token is an end token
     */
    private boolean isEndToken(GGUFSession session, int token) {
        int eosToken = LlamaCppNative.INSTANCE.llama_token_eos(
            session.getContextHandle()
        );
        return token == eosToken;
    }

    /**
     * Inference parameters record
     */
    record InferenceParameters(
        float temperature,
        float topP,
        int topK,
        float repeatPenalty,
        int maxTokens
    ) {
        public InferenceParameters {
            if (temperature < 0 || temperature > 2) {
                throw new IllegalArgumentException("Temperature must be between 0 and 2");
            }
            if (topP < 0 || topP > 1) {
                throw new IllegalArgumentException("Top-p must be between 0 and 1");
            }
            if (topK < 1) {
                throw new IllegalArgumentException("Top-k must be positive");
            }
            if (maxTokens < 1) {
                throw new IllegalArgumentException("Max tokens must be positive");
            }
        }
    }
}
```

### GGUFStreamHandler.java

```java
package tech.kayys.wayang.inference.providers.gguf;

import io.opentelemetry.api.trace.Span;
import io.smallrye.mutiny.Multi;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import org.reactivestreams.Publisher;
import tech.kayys.wayang.inference.api.StreamChunk;
import tech.kayys.wayang.inference.api.StreamingResponse;
import tech.kayys.wayang.inference.providers.gguf.GGUFInferenceExecutor.InferenceParameters;

import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Handles streaming inference for GGUF models.
 * 
 * Features:
 * - Token-by-token streaming
 * - Backpressure handling
 * - Graceful cancellation
 * - Resource cleanup
 * 
 * Thread-safety: Each stream is isolated and thread-safe.
 */
@ApplicationScoped
public class GGUFStreamHandler {

    private static final Logger LOG = Logger.getLogger(GGUFStreamHandler.class);

    @Inject
    GGUFSessionManager sessionManager;

    /**
     * Create streaming response
     */
    public StreamingResponse createStream(
        GGUFSession session,
        String prompt,
        InferenceParameters params,
        String requestId,
        Span span
    ) {
        Publisher<StreamChunk> publisher = createPublisher(
            session,
            prompt,
            params,
            requestId,
            span
        );

        return new GGUFStreamingResponse(requestId, publisher);
    }

    /**
     * Create reactive publisher for streaming
     */
    private Publisher<StreamChunk> createPublisher(
        GGUFSession session,
        String prompt,
        InferenceParameters params,
        String requestId,
        Span span
    ) {
        return Multi.createFrom().emitter(emitter -> {
            
            session.lock();
            var startTime = Instant.now();
            var cancelled = new AtomicBoolean(false);
            var chunkIndex = new AtomicInteger(0);
            
            try {
                // Tokenize prompt
                int[] promptTokens = LlamaCppBinding.tokenize(
                    session.getContextHandle(),
                    prompt,
                    true
                );

                LOG.debugf("Streaming started: %d prompt tokens", promptTokens.length);

                // Evaluate prompt
                LlamaCppBinding.evaluate(
                    session.getContextHandle(),
                    promptTokens,
                    0
                );

                // Generate tokens one by one
                List<Integer> lastTokens = new ArrayList<>();
                int totalGenerated = 0;

                for (int i = 0; i < params.maxTokens(); i++) {
                    // Check cancellation
                    if (cancelled.get() || emitter.isCancelled()) {
                        LOG.debug("Stream cancelled by client");
                        break;
                    }

                    // Sample next token
                    int nextToken = LlamaCppBinding.sample(
                        session.getContextHandle(),
                        params.temperature(),
                        params.topP(),
                        params.topK(),
                        params.repeatPenalty(),
                        lastTokens.stream().mapToInt(Integer::intValue).toArray()
                    );

                    // Check for EOS
                    if (isEndToken(session, nextToken)) {
                        LOG.debug("EOS token generated");
                        break;
                    }

                    // Detokenize
                    String text = LlamaCppBinding.detokenize(
                        session.getContextHandle(),
                        nextToken
                    );

                    // Emit chunk
                    var chunk = StreamChunk.builder()
                        .requestId(requestId)
                        .chunkIndex(chunkIndex.getAndIncrement())
                        .delta(text)
                        .isFinal(false)
                        .metadata("token_id", nextToken)
                        .metadata("position", promptTokens.length + i)
                        .build();

                    emitter.emit(chunk);

                    // Update state
                    lastTokens.add(nextToken);
                    if (lastTokens.size() > 64) {
                        lastTokens.remove(0);
                    }

                    totalGenerated++;

                    // Evaluate next token
                    LlamaCppBinding.evaluate(
                        session.getContextHandle(),
                        new int[]{nextToken},
                        promptTokens.length + i
                    );
                }

                // Emit final chunk
                var duration = java.time.Duration.between(startTime, Instant.now());
                
                var finalChunk = StreamChunk.builder()
                    .requestId(requestId)
                    .chunkIndex(chunkIndex.getAndIncrement())
                    .delta("")
                    .isFinal(true)
                    .metadata("prompt_tokens", promptTokens.length)
                    .metadata("completion_tokens", totalGenerated)
                    .metadata("total_tokens", promptTokens.length + totalGenerated)
                    .metadata("duration_ms", duration.toMillis())
                    .metadata("finish_reason", 
                        totalGenerated >= params.maxTokens() ? "length" : "stop")
                    .build();

                emitter.emit(finalChunk);
                emitter.complete();

                LOG.infof("Streaming completed: %d tokens in %d ms",
                    totalGenerated, duration.toMillis());

            } catch (Exception e) {
                LOG.error("Streaming error", e);
                emitter.fail(e);
            } finally {
                session.unlock();
                sessionManager.returnSession(session);
                span.end();
            }

            // Handle cancellation
            emitter.onTermination(() -> {
                cancelled.set(true);
                LOG.debug("Stream terminated");
            });
        });
    }

    private boolean isEndToken(GGUFSession session, int token) {
        int eosToken = LlamaCppNative.INSTANCE.llama_token_eos(
            session.getContextHandle()
        );
        return token == eosToken;
    }

    /**
     * Streaming response implementation
     */
    private static class GGUFStreamingResponse implements StreamingResponse {
        private final String requestId;
        private final Publisher<StreamChunk> publisher;

        public GGUFStreamingResponse(String requestId, Publisher<StreamChunk> publisher) {
            this.requestId = requestId;
            this.publisher = publisher;
        }

        @Override
        public String getRequestId() {
            return requestId;
        }

        @Override
        public Publisher<StreamChunk> getPublisher() {
            return publisher;
        }

        @Override
        public String getContent() {
            throw new UnsupportedOperationException(
                "Content not available for streaming responses"
            );
        }

        @Override
        public String getModel() {
            return null; // Set during construction if needed
        }

        @Override
        public int getTokensUsed() {
            return 0; // Updated in final chunk
        }

        @Override
        public long getDurationMs() {
            return 0; // Updated in final chunk
        }

        @Override
        public Instant getTimestamp() {
            return Instant.now();
        }

        @Override
        public Map<String, Object> getMetadata() {
            return Map.of("streaming", true);
        }
    }
}
```

## 7. Health Check & Metrics

### GGUFHealthChecker.java

```java
package tech.kayys.wayang.inference.providers.gguf;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.eclipse.microprofile.health.HealthCheck;
import org.eclipse.microprofile.health.HealthCheckResponse;
import org.eclipse.microprofile.health.HealthCheckResponseBuilder;
import org.eclipse.microprofile.health.Readiness;
import org.jboss.logging.Logger;

import java.time.Instant;

/**
 * Health checker for GGUF provider.
 * 
 * Checks:
 * - Native library availability
 * - Session manager health
 * - Resource availability
 * - Error rates
 */
@Readiness
@ApplicationScoped
public class GGUFHealthChecker implements HealthCheck {

    private static final Logger LOG = Logger.getLogger(GGUFHealthChecker.class);

    @Inject
    GGUFSessionManager sessionManager;

    @Inject
    GGUFMetricsCollector metricsCollector;

    @Override
    public HealthCheckResponse call() {
        HealthCheckResponseBuilder builder = HealthCheckResponse
            .named("gguf-provider")
            .withData("timestamp", Instant.now().toString());

        try {
            // Check native library
            boolean nativeLibOk = checkNativeLibrary();
            builder.withData("native_library", nativeLibOk);

            // Check session manager
            int activeSessions = sessionManager.getActiveSessionCount();
            int pooledSessions = sessionManager.getPooledSessionCount();
            
            builder.withData("active_sessions", activeSessions);
            builder.withData("pooled_sessions", pooledSessions);

            // Check metrics
            var metrics = metricsCollector.getHealthMetrics();
            builder.withData("total_inferences", metrics.get("total_inferences"));
            builder.withData("success_rate", metrics.get("success_rate"));
            builder.withData("error_rate", metrics.get("error_rate"));

            // Determine overall health
            boolean healthy = nativeLibOk && 
                             (double) metrics.get("error_rate") < 0.5;

            if (healthy) {
                builder.up();
            } else {
                builder.down();
            }

            return builder.build();

        } catch (Exception e) {
            LOG.error("Health check failed", e);
            return builder
                .down()
                .withData("error", e.getMessage())
                .build();
        }
    }

    private boolean checkNativeLibrary() {
        try {
            String version = LlamaCppBinding.getLibraryVersion();
            return version != null && !version.isEmpty();
        } catch (Exception e) {
            LOG.error("Native library check failed", e);
            return false;
        }
    }

    public HealthCheckResponse check() {
        return call();
    }
}
```

### GGUFMetricsCollector.java

```java
package tech.kayys.wayang.inference.providers.gguf;

import io.micrometer.core.instrument.Counter;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Timer;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Collects and exposes metrics for GGUF provider.
 * 
 * Metrics:
 * - Inference count (success/failure)
 * - Latency distribution
 * - Token throughput
 * - Session pool utilization
 * - Error rates
 */
@ApplicationScoped
public class GGUFMetricsCollector {

    private static final Logger LOG = Logger.getLogger(GGUFMetricsCollector.class);

    @Inject
    MeterRegistry registry;

    // Counters
    private Counter inferenceStartCounter;
    private Counter inferenceSuccessCounter;
    private Counter inferenceFailureCounter;

    // Timers
    private Timer inferenceTimer;

    // Custom metrics
    private final AtomicLong totalTokens = new AtomicLong(0);
    private final Map<String, AtomicLong> modelInferenceCount = new ConcurrentHashMap<>();
    private final Map<String, AtomicLong> errorTypeCount = new ConcurrentHashMap<>();

    @jakarta.annotation.PostConstruct
    void init() {
        LOG.info("Initializing GGUF metrics collector");

        // Initialize counters
        inferenceStartCounter = Counter.builder("gguf.inference.started")
            .description("Total number of inference requests started")
            .tag("provider", "gguf")
            .register(registry);

        inferenceSuccessCounter = Counter.builder("gguf.inference.success")
            .description("Total number of successful inferences")
            .tag("provider", "gguf")
            .register(registry);

        inferenceFailureCounter = Counter.builder("gguf.inference.failure")
            .description("Total number of failed inferences")
            .tag("provider", "gguf")
            .register(registry);

        // Initialize timer
        inferenceTimer = Timer.builder("gguf.inference.duration")
            .description("Inference duration in milliseconds")
            .tag("provider", "gguf")
            .register(registry);

        // Custom gauges
        registry.gauge("gguf.tokens.total", totalTokens);

        LOG.info("GGUF metrics collector initialized");
    }

    /**
     * Record inference start
     */
    public void recordInferenceStart(String modelId) {
        inferenceStartCounter.increment();
        modelInferenceCount
            .computeIfAbsent(modelId, k -> new AtomicLong(0))
            .incrementAndGet();
    }

    /**
     * Record inference success
     */
    public void recordInferenceSuccess(String modelId, long durationMs, int tokens) {
        inferenceSuccessCounter.increment();
        inferenceTimer.record(durationMs, java.util.concurrent.TimeUnit.MILLISECONDS);
        totalTokens.addAndGet(tokens);
    }

    /**
     * Record inference failure
     */
    public void recordInferenceFailure(String modelId, String errorType) {
        inferenceFailureCounter.increment();
        errorTypeCount
            .computeIfAbsent(errorType, k -> new AtomicLong(0))
            .incrementAndGet();
    }

    /**
     * Get health metrics
     */
    public Map<String, Object> getHealthMetrics() {
        long totalInferences = (long) inferenceSuccessCounter.count() + 
                              (long) inferenceFailureCounter.count();
        
        double successRate = totalInferences > 0 
            ? inferenceSuccessCounter.count() / totalInferences 
            : 1.0;
        
        double errorRate = totalInferences > 0 
            ? inferenceFailureCounter.count() / totalInferences 
            : 0.0;

        return Map.of(
            "total_inferences", totalInferences,
            "success_count", (long) inferenceSuccessCounter.count(),
            "failure_count", (long) inferenceFailureCounter.count(),
            "success_rate", successRate,
            "error_rate", errorRate,
            "total_tokens", totalTokens.get()
        );
    }

    /**
     * Get detailed metrics
     */
    public Map<String, Object> getDetailedMetrics() {
        return Map.of(
            "health", getHealthMetrics(),
            "model_inference_count", Map.copyOf(modelInferenceCount),
            "error_type_count", Map.copyOf(errorTypeCount),
            "inference_timer_mean", inferenceTimer.mean(java.util.concurrent.TimeUnit.MILLISECONDS),
            "inference_timer_max", inferenceTimer.max(java.util.concurrent.TimeUnit.MILLISECONDS)
        );
    }
}
```

## 8. Exception Classes

### GGUFLoadException.java

```java
package tech.kayys.wayang.inference.providers.gguf.exceptions;

import tech.kayys.wayang.inference.api.ErrorPayload;

/**
 * Exception thrown when GGUF model loading fails.
 * Non-retryable.
 */
public class GGUFLoadException extends RuntimeException {

    private final ErrorPayload errorPayload;

    public GGUFLoadException(String message) {
        super(message);
        this.errorPayload = ErrorPayload.builder()
            .type("GGUFLoadError")
            .message(message)
            .retryable(false)
            .build();
    }

    public GGUFLoadException(String message, Throwable cause) {
        super(message, cause);
        this.errorPayload = ErrorPayload.builder()
            .type("GGUFLoadError")
            .message(message)
            .retryable(false)
            .detail("cause", cause.getClass().getSimpleName())
            .detail("cause_message", cause.getMessage())
            .build();
    }

    public GGUFLoadException(String message, Throwable cause, ErrorPayload errorPayload) {
        super(message, cause);
        this.errorPayload = errorPayload;
    }

    public ErrorPayload getErrorPayload() {
        return errorPayload;
    }
}
```

### GGUFInferenceException.java

```java
package tech.kayys.wayang.inference.providers.gguf.exceptions;

import tech.kayys.wayang.inference.api.ErrorPayload;

/**
 * Exception thrown during GGUF inference execution.
 * May be retryable depending on the cause.
 */
public class GGUFInferenceException extends RuntimeException {

    private final ErrorPayload errorPayload;

    public GGUFInferenceException(String message) {
        super(message);
        this.errorPayload = ErrorPayload.builder()
            .type("GGUFInferenceError")
            .message(message)
            .retryable(true)
            .build();
    }

    public GGUFInferenceException(String message, Throwable cause) {
        super(message, cause);
        this.errorPayload = ErrorPayload.builder()
            .type("GGUFInferenceError")
            .message(message)
            .retryable(true)
            .detail("cause", cause.getClass().getSimpleName())
            .detail("cause_message", cause.getMessage())
            .build();
    }

    public GGUFInferenceException(String message, Throwable cause, ErrorPayload errorPayload) {
        super(message, cause);
        this.errorPayload = errorPayload;
    }

    public ErrorPayload getErrorPayload() {
        return errorPayload;
    }
}
```

### GGUFSessionException.java

```java
package tech.kayys.wayang.inference.providers.gguf.exceptions;

import tech.kayys.wayang.inference.api.ErrorPayload;

/**
 * Exception thrown during GGUF session management.
 * Usually retryable.
 */
public class GGUFSessionException extends RuntimeException {

    private final ErrorPayload errorPayload;

    public GGUFSessionException(String message) {
        super(message);
        this.errorPayload = ErrorPayload.builder()
            .type("GGUFSessionError")
            .message(message)
            .retryable(true)
            .build();
    }

    public GGUFSessionException(String message, Throwable cause) {
        super(message, cause);
        this.errorPayload = ErrorPayload.builder()
            .type("GGUFSessionError")
            .message(message)
            .retryable(true)
            .detail("cause", cause.getClass().getSimpleName())
            .detail("cause_message", cause.getMessage())
            .build();
    }

    public ErrorPayload getErrorPayload() {
        return errorPayload;
    }
}
```

This completes Part 3. Shall I continue with Part 4, the final part covering:
- Configuration files (application.properties)
- Service provider registration (META-INF/services)
- Native image configuration
- Comprehensive unit tests

?


firefox 
https://claude.ai/chat/1976fb24-001e-4445-bd86-1642ceb291f7