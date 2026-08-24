package tech.kayys.gollek.inference.llamacpp;

import io.micrometer.core.instrument.MeterRegistry;
import io.opentelemetry.api.trace.Tracer;
import io.quarkus.runtime.ShutdownEvent;
import io.quarkus.runtime.StartupEvent;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.event.Observes;
import jakarta.inject.Inject;
import org.eclipse.microprofile.faulttolerance.CircuitBreaker;
import org.eclipse.microprofile.faulttolerance.Timeout;
import org.jboss.logging.Logger;
import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;
import tech.kayys.gollek.spi.inference.LocalInferenceEngine;
import tech.kayys.gollek.spi.exception.ProviderException;
import tech.kayys.gollek.model.repo.local.ManifestStore;
import tech.kayys.gollek.spi.embedding.EmbeddingRequest;
import tech.kayys.gollek.spi.embedding.EmbeddingResponse;
import tech.kayys.gollek.spi.observability.AdapterMetricSchema;
import tech.kayys.gollek.spi.observability.AdapterMetricsRecorder;

import tech.kayys.gollek.spi.observability.NoopAdapterMetricsRecorder;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.Message;
import tech.kayys.alkhawarizm.spi.model.HealthStatus;
import tech.kayys.alkhawarizm.core.tensor.DeviceType;
import tech.kayys.alkhawarizm.core.model.ModelFormat;

import java.time.Duration;
import java.time.Instant;
import java.util.Map;
import java.util.Set;
import java.util.Optional;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Production-ready GGUF inference provider backed by llama.cpp.
 *
 * <p>
 * Implements {@link StreamingProvider} to serve both blocking and streaming
 * inference requests for locally-stored {@code .gguf} model files. Key
 * features:
 * <ul>
 * <li>Lazy initialization — the native llama.cpp backend is loaded on first use
 * unless {@code gguf.provider.prewarm.enabled=true}.</li>
 * <li>Session pooling — {@link LlamaCppSessionManager} maintains a
 * per-tenant/model
 * pool of native contexts with KV-cache reuse across turns.</li>
 * <li>LoRA adapter support — adapters are resolved per-request via
 * {@link LoraAdapterManager} and applied at runtime.</li>
 * <li>Fault tolerance — circuit breaker and timeout annotations guard the
 * {@link #infer} path.</li>
 * <li>Observability — Micrometer counters/timers and OpenTelemetry spans are
 * emitted for every inference call.</li>
 * </ul>
 *
 * <p>
 * Configuration is driven by {@link LlamaCppProviderConfig} (prefix
 * {@code gguf.provider}).
 *
 * @see LlamaCppProviderConfig
 * @see LlamaCppSessionManager
 * @see LoraAdapterManager
 */
@ApplicationScoped
public class LlamaCppEngine implements LocalInferenceEngine {

    private static final Logger log = Logger.getLogger(LlamaCppEngine.class);
    private static final String PROVIDER_ID = "gguf";
    private static final String PROVIDER_VERSION = "1.1.0";

    private final LlamaCppProviderConfig config;
    private final LlamaCppBinding binding;
    private final LlamaCppSessionManager sessionManager;
    private final MeterRegistry meterRegistry;
    private final Tracer tracer;
    private final AdapterMetricsRecorder adapterMetricsRecorder;
    @Inject
    LoraAdapterManager loraAdapterManager;

    private final ManifestStore manifestStore;

    /**
     * Primary CDI constructor.
     *
     * @param config                 provider configuration
     * @param binding                FFM binding to the native llama.cpp library
     * @param sessionManager         manages the pool of native inference contexts
     * @param meterRegistry          Micrometer registry for metrics
     * @param tracer                 OpenTelemetry tracer for distributed tracing
     * @param adapterMetricsRecorder records LoRA adapter acquisition metrics
     * @param manifestStore          manifest store for model resolution
     */
    @Inject
    public LlamaCppEngine(LlamaCppProviderConfig config, LlamaCppBinding binding,
            LlamaCppSessionManager sessionManager,
            MeterRegistry meterRegistry, Tracer tracer, AdapterMetricsRecorder adapterMetricsRecorder,
            ManifestStore manifestStore) {
        this.config = config;
        this.binding = binding;
        this.sessionManager = sessionManager;
        this.meterRegistry = meterRegistry;
        this.tracer = tracer;
        this.adapterMetricsRecorder = adapterMetricsRecorder;
        this.manifestStore = manifestStore;
        initializeStaticMetadata();
    }

    /**
     * Backward-compatible constructor for tests and older wiring paths.
     * Uses a no-op {@link AdapterMetricsRecorder}.
     *
     * @param config         provider configuration
     * @param binding        FFM binding to the native llama.cpp library
     * @param sessionManager manages the pool of native inference contexts
     * @param meterRegistry  Micrometer registry for metrics
     * @param tracer         OpenTelemetry tracer
     */
    public LlamaCppEngine(LlamaCppProviderConfig config, LlamaCppBinding binding,
            LlamaCppSessionManager sessionManager,
            MeterRegistry meterRegistry, Tracer tracer) {
        this(config, binding, sessionManager, meterRegistry, tracer, new NoopAdapterMetricsRecorder(), null);
    }

    private final AtomicBoolean initialized = new AtomicBoolean(false);
    private final AtomicBoolean shutdown = new AtomicBoolean(false);
    private final AtomicBoolean optimizationNoticeLogged = new AtomicBoolean(false);
    private final Map<String, Instant> modelLastUsed = new ConcurrentHashMap<>();

    private void initializeStaticMetadata() {
    }

    @Override
    public void initialize() {
        /*
         * if (!initialized.get()) {
         * if (shutdown.get()) {
         * throw new ProviderException.ProviderInitializationException(
         * PROVIDER_ID,
         * "GGUF provider is shutting down");
         * }
         * throw new ProviderException.ProviderInitializationException(
         * PROVIDER_ID,
         * "GGUF provider failed to initialize during startup");
         * }
         */

        // Initialization is handled lazily via ensureInitialized() on first use,
        // or eagerly via onStart() if prewarm is enabled.
        log.debugf("GGUF provider [%s] initialized in registry", PROVIDER_ID);
    }

    public boolean supports(String modelId, InferenceRequest request) {
        if (!config.enabled()) {
            return false;
        }
        String modelPath = resolveModelPath(modelId);
        if (modelPath == null)
            return false;

        java.nio.file.Path p = java.nio.file.Paths.get(modelPath);
        boolean exists = java.nio.file.Files.exists(p);

        // v2.2.0 Hardening: Verify magic header if it's a file
        if (exists && java.nio.file.Files.isRegularFile(p)) {
            if (!tech.kayys.alkhawarizm.spi.model.ModelFormatDetector.isGguf(p)) {
                log.debugf("Model rejected: %s (not a GGUF file)", modelPath);
                return false;
            }
        }

        log.debugf("Model support check: %s -> %s (exists=%b)", modelId, modelPath, exists);
        return exists;
    }

    @Override
    @CircuitBreaker(requestVolumeThreshold = 10, failureRatio = 0.5, delay = 30000, successThreshold = 3)
    @Timeout(value = 30, unit = java.time.temporal.ChronoUnit.SECONDS)
    public Uni<InferenceResponse> infer(InferenceRequest request) {
        ensureInitialized();

        String tenantId = resolveTenantId(request);
        var span = tracer.spanBuilder("gguf.infer")
                .setAttribute("model", request.getModel())
                .setAttribute("tenant", tenantId)
                .startSpan();

        return Uni.createFrom().item(() -> {
            LlamaCppSessionManager.SessionContext sessionContext = null;
            AdapterSpec adapterSpec = null;
            try {
                Instant startTime = Instant.now();

                log.debugf("Starting inference for model=%s, tenant=%s",
                        request.getModel(), tenantId);

                adapterSpec = resolveAdapterSpec(request, tenantId);
                InferenceRequest inferenceRequest = convertToInferenceRequest(request, adapterSpec);
                Instant adapterAcquireStart = Instant.now();

                sessionContext = sessionManager.getSession(
                        tenantId,
                        request.getModel(),
                        config,
                        adapterSpec);
                if (adapterSpec != null) {
                    recordAdapterMetrics(adapterSpec, Duration.between(adapterAcquireStart, Instant.now()));
                }

                if (sessionContext == null) {
                    throw new IllegalStateException(
                            "Failed to acquire session context for model: " + request.getModel());
                }

                InferenceResponse response = sessionContext.runner().infer(inferenceRequest);
                recordMicrometerMetrics(request.getModel(), true, Duration.between(startTime, Instant.now()));

                return response;

            } catch (Exception e) {
                recordMicrometerMetrics(request.getModel(), false, Duration.ZERO);

                String message = e.getMessage() == null ? "" : e.getMessage();
                if (message.contains("timed out")) {
                    log.warnf("Inference timeout for model=%s: %s", request.getModel(), message);
                } else {
                    log.errorf(e, "Inference failed for model=%s", request.getModel());
                }

                throw new RuntimeException("Inference failed: " + e.getMessage(), e);
            } finally {
                if (sessionContext != null) {
                    sessionManager.releaseSession(
                            tenantId,
                            request.getModel(),
                            sessionContext,
                            adapterSpec);
                }
                span.end();
            }
        })
                .runSubscriptionOn(io.smallrye.mutiny.infrastructure.Infrastructure.getDefaultWorkerPool());
    }

    @Override
    public Uni<EmbeddingResponse> executeEmbedding(String modelId, EmbeddingRequest request) {
        ensureInitialized();

        return Uni.createFrom().deferred(() -> {
            LlamaCppSessionManager.SessionContext sessionContext = null;
            String tenantId = resolveTenantId(request.model(), request.parameters(), Optional.empty());
            try {
                sessionContext = sessionManager.getSession(
                        tenantId,
                        request.model(),
                        config);

                if (sessionContext == null) {
                    return Uni.createFrom().failure(new IllegalStateException(
                            "Failed to acquire session context for model: " + request.model()));
                }

                // Final context for closure
                final LlamaCppSessionManager.SessionContext finalSession = sessionContext;

                return sessionContext.runner().embed(request)
                        .onTermination().invoke(() -> {
                            sessionManager.releaseSession(
                                    tenantId,
                                    request.model(),
                                    finalSession,
                                    null);
                        });

            } catch (Exception e) {
                log.errorf(e, "Embedding failed for model=%s", request.model());
                return Uni.createFrom().failure(new RuntimeException("Embedding failed: " + e.getMessage(), e));
            }
        });
    }

    private String resolveTenantId(String model, Map<String, Object> metadata, Optional<String> userId) {
        // Try metadata "tenantId"
        Object tenantId = metadata.get("tenantId");
        if (tenantId instanceof String && !((String) tenantId).isBlank()) {
            return (String) tenantId;
        }
        // Try userId
        if (userId.isPresent()) {
            return userId.get();
        }
        return "community";
    }

    @Override
    public Multi<StreamingInferenceChunk> stream(InferenceRequest request) {
        ensureInitialized();

        String tenantId = resolveTenantId(request);

        return Multi.createFrom().deferred(() -> {
            LlamaCppSessionManager.SessionContext sessionContext = null;
            try {
                log.debugf("Starting streaming inference for model=%s, tenant=%s",
                        request.getModel(), tenantId);

                AdapterSpec adapterSpec = resolveAdapterSpec(request, tenantId);
                InferenceRequest inferenceRequest = convertToInferenceRequest(request, adapterSpec);

                // Determine hardware acceleration info
                boolean gpuActive = LlamaCppDeviceSupport.effectiveGpuEnabled(config);
                String hardware = gpuActive ? "GPU/Metal" : "CPU";
                int gpuLayers = LlamaCppDeviceSupport.resolveGpuLayers(config);
                if (gpuActive && gpuLayers != 0) {
                    hardware += " (layers: " + (gpuLayers == -1 ? "all" : gpuLayers) + ")";
                }

                final Map<String, Object> metadata = Map.of("hardware", hardware);

                Instant adapterAcquireStart = Instant.now();

                sessionContext = sessionManager.getSession(
                        tenantId,
                        request.getModel(),
                        config,
                        adapterSpec);
                if (adapterSpec != null) {
                    recordAdapterMetrics(adapterSpec, Duration.between(adapterAcquireStart, Instant.now()));
                }

                if (sessionContext == null) {
                    return Multi.createFrom().failure(new IllegalStateException(
                            "Failed to acquire session context for model: " + request.getModel()));
                }

                // Final session context for closure
                final LlamaCppSessionManager.SessionContext finalSession = sessionContext;

                AtomicBoolean first = new AtomicBoolean(true);
                return sessionContext.runner().inferStream(
                        inferenceRequest)
                        .map(chunk -> {
                            if (first.compareAndSet(true, false)) {
                                return new StreamingInferenceChunk(
                                        chunk.requestId(),
                                        chunk.index(),
                                        chunk.modality(),
                                        chunk.delta(),
                                        chunk.imageDeltaBase64(),
                                        chunk.finished(),
                                        chunk.finishReason(),
                                        chunk.usage(),
                                        chunk.emittedAt(),
                                        metadata,
                                        null,
                                        null,
                                        null);
                            }
                            return chunk;
                        })
                        .onTermination().invoke(() -> {
                            sessionManager.releaseSession(
                                    tenantId,
                                    request.getModel(),
                                    finalSession,
                                    adapterSpec);
                        });

            } catch (Exception e) {
                log.errorf(e, "Streaming inference acquisition failed for model=%s", request.getModel());
                return Multi.createFrom().failure(new RuntimeException("Streaming inference failed: " + e.getMessage(), e));
            }
        });
    }

    private String resolveTenantId(InferenceRequest request) {
        // Try metadata "tenantId"
        Object tenantId = request.getMetadata().get("tenantId");
        if (tenantId instanceof String && !((String) tenantId).isBlank()) {
            return (String) tenantId;
        }
        // Try userId
        if (request.getUserId().isPresent()) {
            return request.getUserId().get();
        }
        return "community";
    }

    @Override
    public HealthStatus health() {
        try {
            if (!initialized.get()) {
                return HealthStatus.unhealthy("GGUF Provider not initialized");
            }
            if (sessionManager != null && !sessionManager.isHealthy()) {
                return HealthStatus.unhealthy("GGUF session manager unhealthy");
            }
            return HealthStatus.healthy("GGUF Engine OK");
        } catch (Throwable t) {
            log.error("Error checking GGUF Provider health", t);
            return HealthStatus.unhealthy(String.valueOf(t.getMessage()));
        }
    }

    @Override
    public void shutdown() {
        if (shutdown.compareAndSet(false, true)) {
            log.info("Shutting down GGUF Provider");
            if (initialized.get()) {
                try {
                    sessionManager.shutdown();
                } catch (Throwable t) {
                    log.warn("Session manager shutdown failed", t);
                }
                try {
                    // In native image teardown we avoid backend_free because process teardown
                    // can race with native worker cleanup and produce exit-133 crashes.
                    if (!isNativeImageRuntime()) {
                        binding.backendFree();
                    } else {
                        log.debug("Skipping llama backendFree on native-image shutdown");
                    }
                } catch (Throwable t) {
                    log.warn("llama backend shutdown failed", t);
                } finally {
                    modelLastUsed.clear();
                    initialized.set(false);
                }
            }
        }
    }

    private boolean isNativeImageRuntime() {
        return System.getProperty("org.graalvm.nativeimage.imagecode") != null;
    }

    private Set<DeviceType> buildSupportedDevices() {
        return LlamaCppDeviceSupport.supportedDevices(config);
    }

    private void prewarmModels(java.util.List<String> modelIds) {
        modelIds.forEach(modelId -> {
            if (supports(modelId, null)) {
                try {
                    sessionManager.getSession("system", modelId, config);
                } catch (Exception e) {
                    log.warn("Prewarm failed", e);
                }
            }
        });
    }

    private String resolveModelPath(String modelId) {
        if (modelId == null)
            return null;
        if (modelId.startsWith("/"))
            return modelId;

        // 1. Try manifest resolution
        Optional<tech.kayys.gollek.model.repo.local.GollekManifest> manifest = manifestStore.findByModelId(modelId,
                "gguf");
        if (manifest.isPresent()) {
            return manifest.get().getBlobPath();
        }

        // 2. Fallback to name-based lookup
        String normalizedId = modelId.replace("/", "__").replace(".", "_");
        manifest = manifestStore.findByName(normalizedId);
        if (manifest.isPresent()) {
            return manifest.get().getBlobPath();
        }

        log.warnf("GGUF model not found in manifest: %s. Falling back to default path calculation.", modelId);
        return ManifestStore.getModelsRoot().resolve("blobs").resolve(normalizedId).toAbsolutePath().toString();
    }

    private InferenceRequest convertToInferenceRequest(InferenceRequest request, AdapterSpec adapterSpec) {
        // Apply Chat Template (Defaulting to ChatML for now as Qwen uses it)
        // TODO: detect template type from model metadata or config
        String prompt = applyChatMLTemplate(request.getMessages());

        var builder = InferenceRequest.builder()
                .model(request.getModel())
                .messages(request.getMessages())
                .parameter("prompt", prompt)
                .parameter("max_tokens", Math.max(16, request.getMaxTokens()))
                .parameter("temperature", request.getTemperature())
                .parameter("top_p", request.getTopP())
                .parameter("top_k", request.getTopK() != 0 ? request.getTopK() : config.defaultTopK())
                .parameter("json_mode", request.isJsonMode() || config.defaultJsonMode());

        // Add additional sampling parameters from provider request
        request.getParameters().forEach((k, v) -> {
            if (!k.equals("prompt") && !k.equals("max_tokens") && !k.equals("temperature") && !k.equals("top_p")
                    && !k.equals("top_k")) {
                builder.parameter(k, v);
            }
        });

        if (adapterSpec != null) {
            builder.parameter("adapter_type", adapterSpec.type());
            builder.parameter("adapter_id", adapterSpec.adapterId());
            builder.parameter("adapter_path", adapterSpec.adapterPath());
            builder.parameter("adapter_scale", adapterSpec.scale());
            // Keep aliases for backward compatibility with provider-specific code paths.
            builder.parameter("lora_adapter_id", adapterSpec.adapterId());
            builder.parameter("lora_adapter_path", adapterSpec.adapterPath());
            builder.parameter("lora_scale", adapterSpec.scale());
        }

        return builder.build();
    }

    private AdapterSpec resolveAdapterSpec(InferenceRequest request, String tenantId) {
        if (loraAdapterManager == null) {
            return null;
        }
        return loraAdapterManager.resolve(request, tenantId).orElse(null);
    }

    private String applyChatMLTemplate(java.util.List<Message> messages) {
        StringBuilder sb = new StringBuilder();
        for (Message msg : messages) {
            sb.append("<|im_start|>").append(msg.getRole().toString().toLowerCase()).append("\n");
            sb.append(msg.getContent()).append("<|im_end|>\n");
        }
        // Append generation prompt
        sb.append("<|im_start|>assistant\n");
        return sb.toString();
    }

    private void recordMicrometerMetrics(String modelId, boolean success, Duration duration) {
        if (config.metricsEnabled()) {
            meterRegistry.counter("gguf.inference.total", "model", modelId, "status", success ? "success" : "failure")
                    .increment();
            if (success) {
                meterRegistry.timer("gguf.inference.duration", "model", modelId).record(duration);
            }
        }
    }

    private void recordAdapterMetrics(AdapterSpec adapterSpec, Duration acquireDuration) {
        if (!config.metricsEnabled() || adapterSpec == null) {
            return;
        }
        var schema = AdapterMetricSchema.builder()
                .adapterId(adapterSpec.adapterId())
                .operation("session_acquire")
                .build();
        adapterMetricsRecorder.recordSuccess(schema, acquireDuration.toMillis());
    }

    private boolean isRetryable(Throwable error) {
        if (error instanceof java.util.concurrent.TimeoutException)
            return true;
        if (error instanceof OutOfMemoryError)
            return false;
        String msg = error.getMessage();
        return msg == null || (!msg.toLowerCase().contains("not found") && !msg.toLowerCase().contains("invalid"));
    }

    private synchronized void ensureInitialized() {
        if (shutdown.get()) {
            throw new IllegalStateException("GGUF Provider has been shutdown");
        }
        if (initialized.get()) {
            return;
        }

        if (!config.enabled()) {
            log.info("GGUF Provider is disabled by configuration");
            // Mark as initialized but unusable? Or just return and let usages fail?
            // If disabled, initialized remains false.
            throw new IllegalStateException("GGUF Provider is disabled by configuration");
        }

        log.infof("Initializing GGUF Provider v%s", PROVIDER_VERSION);

        try {
            initializeStaticMetadata();

            if (LlamaCppDeviceSupport.shouldAutoMetal(config)) {
                int autoLayers = LlamaCppDeviceSupport.resolveGpuLayers(config);
                log.infof("GGUF auto-Metal enabled (layers=%d)", autoLayers);
            }

            if (LlamaCppOptimizationDetector.hasOptimizationModules()
                    && optimizationNoticeLogged.compareAndSet(false, true)) {
                log.info(
                        "GGUF optimization modules detected on classpath; GGUF uses llama.cpp and only advertises availability.");
            }

            log.debug("Initializing GGUF native backend");
            if (binding != null) {
                binding.backendInit();
                log.info("llama.cpp native library initialized");
            } else {
                log.warn("GGUF native binding is not available; llama.cpp backend disabled.");
                throw new IllegalStateException("Native binding not available");
            }

            sessionManager.initialize();

            if (config.prewarmEnabled() && config.prewarmModels().isPresent()) {
                prewarmModels(config.prewarmModels().get());
            }

            initialized.set(true);
            log.info("GGUF Provider initialization completed");

        } catch (Throwable t) {
            if (config.verboseLogging()) {
                log.warn("GGUF Provider failed to initialize (native library may be missing or incompatible): "
                        + t.getMessage());
                log.debug("Initialization error details:", t);
            } else {
                log.debug("GGUF Provider failed to initialize: " + t.getMessage(), t);
            }
            // We don't throw an exception here to allow the rest of the application to
            // start
            // and use other providers (e.g. safetensor).
            initialized.set(false);
        }
    }
}
