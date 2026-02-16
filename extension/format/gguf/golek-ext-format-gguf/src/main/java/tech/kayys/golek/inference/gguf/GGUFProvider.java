package tech.kayys.golek.inference.gguf;

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
import tech.kayys.golek.spi.stream.StreamChunk;

import tech.kayys.golek.spi.exception.ProviderException;
import tech.kayys.golek.spi.inference.InferenceRequest;
import tech.kayys.golek.spi.inference.InferenceResponse;
import tech.kayys.golek.spi.model.DeviceType;
import tech.kayys.golek.spi.model.ModelFormat;
import tech.kayys.golek.spi.provider.*;
import tech.kayys.golek.spi.Message;

import java.time.Duration;
import java.time.Instant;
import java.util.Map;
import java.util.Set;
import java.util.Optional;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Production-ready GGUF provider implementation using llama.cpp.
 */
@ApplicationScoped

public class GGUFProvider implements StreamingProvider {

    private static final Logger log = Logger.getLogger(GGUFProvider.class);
    private static final String PROVIDER_ID = "gguf";
    private static final String PROVIDER_VERSION = "1.1.0";

    private final GGUFProviderConfig config;
    private final LlamaCppBinding binding;
    private final GGUFSessionManager sessionManager;
    private final MeterRegistry meterRegistry;
    private final Tracer tracer;

    @Inject
    public GGUFProvider(GGUFProviderConfig config, LlamaCppBinding binding, GGUFSessionManager sessionManager,
            MeterRegistry meterRegistry, Tracer tracer) {
        this.config = config;
        this.binding = binding;
        this.sessionManager = sessionManager;
        this.meterRegistry = meterRegistry;
        this.tracer = tracer;
        initializeStaticMetadata();
    }

    private final AtomicBoolean initialized = new AtomicBoolean(false);
    private final AtomicBoolean shutdown = new AtomicBoolean(false);
    private final ProviderMetrics metrics = new ProviderMetrics();
    private final Map<String, Instant> modelLastUsed = new ConcurrentHashMap<>();

    private ProviderMetadata metadata;
    private ProviderCapabilities capabilities;

    void onStart(@Observes StartupEvent event) {
        // Keep provider lazy by default to avoid loading GGUF runtime when not
        // selected.
        if (!config.prewarmEnabled()) {
            return;
        }
        try {
            ensureInitialized();
        } catch (Exception e) {
            log.warn("GGUF prewarm initialization failed (will retry lazily): " + e.getMessage());
        }
    }

    void onStop(@Observes ShutdownEvent event) {
        shutdown();
    }

    @Override
    public String id() {
        return PROVIDER_ID;
    }

    @Override
    public String name() {
        return "GGUF Provider (llama.cpp)";
    }

    @Override
    public String version() {
        return PROVIDER_VERSION;
    }

    @Override
    public ProviderMetadata metadata() {
        return metadata;
    }

    @Override
    public ProviderCapabilities capabilities() {
        return capabilities;
    }

    private void initializeStaticMetadata() {
        if (this.metadata == null) {
            this.metadata = ProviderMetadata.builder()
                    .providerId(PROVIDER_ID)
                    .name("GGUF Provider (llama.cpp)")
                    .version(PROVIDER_VERSION)
                    .description("Local GGUF model inference using llama.cpp")
                    .vendor("Kayys Tech")
                    .homepage("https://github.com/ggerganov/llama.cpp")
                    .build();
        }

        if (this.capabilities == null) {
            this.capabilities = ProviderCapabilities.builder()
                    .streaming(true)
                    .functionCalling(false)
                    .multimodal(false)
                    .toolCalling(false)
                    .embeddings(false)
                    .maxContextTokens(config.maxContextTokens())
                    .maxOutputTokens(config.maxContextTokens() / 2)
                    .supportedFormats(Set.of(ModelFormat.GGUF))
                    .supportedDevices(buildSupportedDevices())
                    .supportedLanguages(java.util.List.of("en"))
                    .features(Set.of("local_inference", "cpu_inference",
                            config.gpuEnabled() ? "gpu_acceleration" : "cpu_only"))
                    .build();
        }
    }

    @Override
    public void initialize(ProviderConfig config) throws ProviderException.ProviderInitializationException {
        if (!initialized.get()) {
            if (shutdown.get()) {
                throw new ProviderException.ProviderInitializationException(
                        PROVIDER_ID,
                        "GGUF provider is shutting down",
                        null);
            }
            throw new ProviderException.ProviderInitializationException(
                    PROVIDER_ID,
                    "GGUF provider failed to initialize during startup",
                    null);
        }
    }

    @Override
    public boolean supports(String modelId, ProviderRequest request) {
        String modelPath = resolveModelPath(modelId);
        if (modelPath == null)
            return false;
        boolean exists = java.nio.file.Files.exists(java.nio.file.Paths.get(modelPath));
        log.debugf("Model support check: %s -> %s (exists=%b)", modelId, modelPath, exists);
        return exists;
    }

    @Override
    @CircuitBreaker(requestVolumeThreshold = 10, failureRatio = 0.5, delay = 30000, successThreshold = 3)
    @Timeout(value = 30, unit = java.time.temporal.ChronoUnit.SECONDS)
    public Uni<InferenceResponse> infer(ProviderRequest request) {
        ensureInitialized();

        String tenantId = resolveTenantId(request);
        var span = tracer.spanBuilder("gguf.infer")
                .setAttribute("model", request.getModel())
                .setAttribute("tenant", tenantId)
                .startSpan();

        return Uni.createFrom().item(() -> {
            GGUFSessionManager.SessionContext sessionContext = null;
            try {
                metrics.recordRequest();
                Instant startTime = Instant.now();

                log.debugf("Starting inference for model=%s, tenant=%s",
                        request.getModel(), tenantId);

                InferenceRequest inferenceRequest = convertToInferenceRequest(request);

                sessionContext = sessionManager.getSession(
                        tenantId,
                        request.getModel(),
                        config);

                if (sessionContext == null) {
                    throw new IllegalStateException(
                            "Failed to acquire session context for model: " + request.getModel());
                }

                InferenceResponse response = sessionContext.runner().infer(inferenceRequest);
                metrics.recordSuccess();
                recordMicrometerMetrics(request.getModel(), true, Duration.between(startTime, Instant.now()));

                return response;

            } catch (Exception e) {
                metrics.recordFailure();
                recordMicrometerMetrics(request.getModel(), false, Duration.ZERO);

                log.errorf(e, "Inference failed for model=%s", request.getModel());

                throw new ProviderException(
                        PROVIDER_ID,
                        "Inference failed: " + e.getMessage(),
                        e,
                        isRetryable(e));
            } finally {
                if (sessionContext != null) {
                    sessionManager.releaseSession(
                            tenantId,
                            request.getModel(),
                            sessionContext);
                }
                span.end();
            }
        })
                .runSubscriptionOn(io.smallrye.mutiny.infrastructure.Infrastructure.getDefaultWorkerPool());
    }

    @Override
    public Multi<StreamChunk> inferStream(ProviderRequest request) {
        ensureInitialized();

        String tenantId = resolveTenantId(request);

        return Multi.createFrom().deferred(() -> {
            GGUFSessionManager.SessionContext sessionContext = null;
            try {
                metrics.recordRequest();

                log.debugf("Starting streaming inference for model=%s, tenant=%s",
                        request.getModel(), tenantId);

                InferenceRequest inferenceRequest = convertToInferenceRequest(request);

                sessionContext = sessionManager.getSession(
                        tenantId,
                        request.getModel(),
                        config);

                if (sessionContext == null) {
                    return Multi.createFrom().failure(new IllegalStateException(
                            "Failed to acquire session context for model: " + request.getModel()));
                }

                // Final session context for closure
                final GGUFSessionManager.SessionContext finalSession = sessionContext;

                return sessionContext.runner().inferStream(
                        inferenceRequest)
                        .onTermination().invoke(() -> {
                            sessionManager.releaseSession(
                                    tenantId,
                                    request.getModel(),
                                    finalSession);
                        });

            } catch (Exception e) {
                metrics.recordFailure();
                log.errorf(e, "Streaming inference acquisition failed for model=%s", request.getModel());
                return Multi.createFrom().failure(new ProviderException(
                        PROVIDER_ID,
                        "Streaming inference failed: " + e.getMessage(),
                        e,
                        isRetryable(e)));
            }
        });
    }

    private String resolveTenantId(ProviderRequest request) {
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
    public Uni<ProviderHealth> health() {
        return Uni.createFrom().item(() -> {
            try {
                var status = initialized.get() ? ProviderHealth.Status.HEALTHY : ProviderHealth.Status.DEGRADED;
                var details = new java.util.HashMap<String, Object>();

                details.put("initialized", initialized.get());
                if (metadata != null) {
                    details.put("version", metadata.getVersion());
                }

                if (sessionManager != null) {
                    details.put("active_sessions", sessionManager.getActiveSessionCount());
                    if (!sessionManager.isHealthy()) {
                        status = ProviderHealth.Status.DEGRADED;
                        details.put("session_manager", "degraded");
                    }
                }

                if (config.maxMemoryBytes() > 0) {
                    long usedMemory = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
                    if (usedMemory > config.maxMemoryBytes() * 0.9) {
                        status = ProviderHealth.Status.DEGRADED;
                        details.put("memory_warning", "High memory usage");
                    }
                }

                if (metrics != null && metrics.getTotalRequests() > 10) {
                    double failureRate = 1.0 - metrics.getSuccessRate();
                    details.put("failure_rate", String.format("%.2f%%", failureRate * 100));

                    if (failureRate > 0.5)
                        status = ProviderHealth.Status.UNHEALTHY;
                    else if (failureRate > 0.2)
                        status = ProviderHealth.Status.DEGRADED;
                }

                return ProviderHealth.builder()
                        .status(status)
                        .details(details)
                        .timestamp(Instant.now())
                        .build();

            } catch (Throwable t) {
                log.error("Error checking GGUF Provider health", t);
                return ProviderHealth.builder()
                        .status(ProviderHealth.Status.UNHEALTHY)
                        .details(java.util.Map.of("error", String.valueOf(t.getMessage())))
                        .timestamp(Instant.now())
                        .build();
            }
        });
    }

    @Override
    public Optional<ProviderMetrics> metrics() {
        return Optional.of(metrics);
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
        return config.gpuEnabled() ? Set.of(DeviceType.CPU, DeviceType.CUDA) : Set.of(DeviceType.CPU);
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

        String normalizedId = modelId.replace("/", "_");
        java.util.List<java.nio.file.Path> modelDirs = new java.util.ArrayList<>();
        try {
            modelDirs.add(java.nio.file.Paths.get(config.modelBasePath()));
        } catch (Exception ignored) {
            // ignore invalid configured path
        }
        modelDirs.add(java.nio.file.Paths.get(System.getProperty("user.home"), ".golek", "models", "gguf"));
        modelDirs = modelDirs.stream()
                .map(java.nio.file.Path::toAbsolutePath)
                .map(java.nio.file.Path::normalize)
                .distinct()
                .toList();

        // Try variations
        String[] variations = {
                normalizedId,
                normalizedId + ".gguf",
                normalizedId + "-GGUF",
                normalizedId + "-GGUF.gguf",
                modelId,
                modelId + ".gguf",
                modelId + "-GGUF"
        };

        for (java.nio.file.Path modelDir : modelDirs) {
            for (String var : variations) {
                java.nio.file.Path p = modelDir.resolve(var);
                if (java.nio.file.Files.exists(p)) {
                    return p.toString();
                }
            }
        }

        java.nio.file.Path defaultPath = modelDirs.get(0).resolve(normalizedId);
        return defaultPath.toAbsolutePath().toString();
    }

    private InferenceRequest convertToInferenceRequest(ProviderRequest request) {
        // Apply Chat Template (Defaulting to ChatML for now as Qwen uses it)
        // TODO: detect template type from model metadata or config
        String prompt = applyChatMLTemplate(request.getMessages());

        var builder = InferenceRequest.builder()
                .model(request.getModel())
                .messages(request.getMessages())
                .parameter("prompt", prompt)
                .parameter("max_tokens", Math.min(128, Math.max(16, request.getMaxTokens())))
                .parameter("temperature", request.getTemperature())
                .parameter("top_p", request.getTopP())
                .parameter("top_k", request.getParameter("top_k", Integer.class).orElse(config.defaultTopK()))
                .parameter("stop", List.of("<|im_end|>", "<|endoftext|>", "</s>"))
                .parameter("json_mode",
                        request.getParameter("json_mode", Boolean.class).orElse(config.defaultJsonMode()));

        // Add additional sampling parameters from provider request
        request.getParameters().forEach((k, v) -> {
            if (!k.equals("prompt") && !k.equals("max_tokens") && !k.equals("temperature") && !k.equals("top_p")
                    && !k.equals("top_k")) {
                builder.parameter(k, v);
            }
        });

        return builder.build();
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

            log.debug("Initializing GGUF native backend");
            binding.backendInit();
            log.info("llama.cpp native library initialized");

            sessionManager.initialize();

            if (config.prewarmEnabled() && config.prewarmModels().isPresent()) {
                prewarmModels(config.prewarmModels().get());
            }

            initialized.set(true);
            log.info("GGUF Provider initialization completed");

        } catch (Throwable t) {
            t.printStackTrace();
            System.err.println("FATAL: GGUF Provider initialization failed: " + t.getMessage());
            log.error("Failed to initialize GGUF Provider", t);
            throw new IllegalStateException("Failed to initialize GGUF Provider: " + t.getMessage(), t);
        }
    }
}
