package tech.kayys.golek.inference.gguf;

import io.micrometer.core.instrument.MeterRegistry;
import io.opentelemetry.api.trace.Tracer;
import io.quarkus.runtime.ShutdownEvent;
import io.quarkus.runtime.StartupEvent;

import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.event.Observes;
import jakarta.inject.Inject;
import org.eclipse.microprofile.faulttolerance.CircuitBreaker;
import org.eclipse.microprofile.faulttolerance.Timeout;
import org.jboss.logging.Logger;

import tech.kayys.golek.spi.exception.ProviderException;
import tech.kayys.golek.spi.inference.InferenceRequest;
import tech.kayys.golek.spi.inference.InferenceResponse;
import tech.kayys.golek.spi.model.DeviceType;
import tech.kayys.golek.spi.model.ModelFormat;
import tech.kayys.golek.spi.provider.*;
import tech.kayys.wayang.tenant.TenantContext;
import tech.kayys.golek.spi.Message;

import java.time.Duration;
import java.time.Instant;
import java.util.Map;
import java.util.Set;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.Collectors;

/**
 * Production-ready GGUF provider implementation using llama.cpp.
 */
@ApplicationScoped

public class GGUFProvider implements LLMProvider {

    private static final Logger log = Logger.getLogger(GGUFProvider.class);
    private static final String PROVIDER_ID = "gguf-llama-cpp";
    private static final String PROVIDER_VERSION = "1.1.0";

    @Inject
    GGUFProviderConfig config;

    @Inject
    LlamaCppBinding binding;

    @Inject
    GGUFSessionManager sessionManager;

    @Inject
    MeterRegistry meterRegistry;

    @Inject
    Tracer tracer;

    private final AtomicBoolean initialized = new AtomicBoolean(false);
    private final ProviderMetrics metrics = new ProviderMetrics();
    private final Map<String, Instant> modelLastUsed = new ConcurrentHashMap<>();

    private ProviderMetadata metadata;
    private ProviderCapabilities capabilities;

    void onStart(@Observes StartupEvent event) {
        log.infof("Initializing GGUF Provider v%s", PROVIDER_VERSION);

        try {
            binding.backendInit();
            log.info("llama.cpp native library initialized");

            sessionManager.initialize();

            this.metadata = ProviderMetadata.builder()
                    .providerId(PROVIDER_ID)
                    .name("GGUF Provider (llama.cpp)")
                    .version(PROVIDER_VERSION)
                    .description("Local GGUF model inference using llama.cpp")
                    .vendor("Kayys Tech")
                    .homepage("https://github.com/ggerganov/llama.cpp")
                    .build();

            this.capabilities = ProviderCapabilities.builder()
                    .streaming(false)
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

            if (config.prewarmEnabled() && config.prewarmModels().isPresent()) {
                prewarmModels(config.prewarmModels().get());
            }

            initialized.set(true);
            log.info("GGUF Provider initialization completed");

        } catch (Exception e) {
            log.error("Failed to initialize GGUF Provider", e);
            throw new IllegalStateException("GGUF Provider initialization failed", e);
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

    @Override
    public void initialize(ProviderConfig config) throws ProviderException.ProviderInitializationException {
        if (!initialized.get()) {
            throw new ProviderException.ProviderInitializationException(
                    PROVIDER_ID,
                    "GGUF provider failed to initialize during startup",
                    null);
        }
    }

    @Override
    public boolean supports(String modelId, TenantContext tenantContext) {
        String modelPath = resolveModelPath(modelId);
        boolean exists = java.nio.file.Files.exists(java.nio.file.Paths.get(modelPath));
        log.debugf("Model support check: %s -> %s (exists=%b)", modelId, modelPath, exists);
        return exists;
    }

    @Override
    @CircuitBreaker(requestVolumeThreshold = 10, failureRatio = 0.5, delay = 30000, successThreshold = 3)
    @Timeout(value = 30, unit = java.time.temporal.ChronoUnit.SECONDS)
    public Uni<InferenceResponse> infer(ProviderRequest request, TenantContext context) {
        ensureInitialized();

        TenantContext effectiveContext = ensureTenantContext(context);
        var span = tracer.spanBuilder("gguf.infer")
                .setAttribute("model", request.getModel())
                .setAttribute("tenant", effectiveContext.getTenantId().value())
                .startSpan();

        return Uni.createFrom().item(() -> {
            try {
                metrics.recordRequest();
                Instant startTime = Instant.now();

                log.debugf("Starting inference for model=%s, tenant=%s",
                        request.getModel(), effectiveContext.getTenantId().value());

                InferenceRequest inferenceRequest = convertToInferenceRequest(request);

                var sessionContext = sessionManager.getSession(
                        effectiveContext.getTenantId().value(),
                        request.getModel(),
                        config);

                if (sessionContext == null) {
                    throw new IllegalStateException(
                            "Failed to acquire session context for model: " + request.getModel());
                }

                InferenceResponse response = sessionContext.runner().infer(
                        inferenceRequest,
                        createRequestContext(effectiveContext, request));

                long durationMs = Duration.between(startTime, Instant.now()).toMillis();
                metrics.recordSuccess();
                metrics.recordDuration(durationMs);

                modelLastUsed.put(request.getModel(), Instant.now());
                recordMicrometerMetrics(request.getModel(), true, Duration.ofMillis(durationMs));

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
                span.end();
            }
        });
    }

    @Override
    public Uni<ProviderHealth> health() {
        return Uni.createFrom().item(() -> {
            try {
                var status = ProviderHealth.Status.HEALTHY;
                var details = new java.util.HashMap<String, Object>();

                details.put("initialized", initialized.get());
                details.put("total_inferences", metrics.getTotalRequests());
                details.put("failed_inferences", metrics.getFailedRequests());
                details.put("active_sessions", sessionManager.getActiveSessionCount());

                if (!sessionManager.isHealthy()) {
                    status = ProviderHealth.Status.DEGRADED;
                    details.put("session_manager", "degraded");
                }

                if (config.maxMemoryBytes() > 0) {
                    long usedMemory = Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory();
                    if (usedMemory > config.maxMemoryBytes() * 0.9) {
                        status = ProviderHealth.Status.DEGRADED;
                        details.put("memory_warning", "High memory usage");
                    }
                }

                if (metrics.getTotalRequests() > 10) {
                    double failureRate = 1.0 - metrics.getSuccessRate();
                    details.put("failure_rate", String.format("%.2f%%", failureRate * 100));

                    if (failureRate > 0.5)
                        status = ProviderHealth.Status.UNHEALTHY;
                    else if (failureRate > 0.2)
                        status = ProviderHealth.Status.DEGRADED;
                }

                return ProviderHealth.builder()
                        .status(status)
                        .message(status == ProviderHealth.Status.HEALTHY ? "Healthy" : "Degraded/Unhealthy")
                        .details(details)
                        .timestamp(Instant.now())
                        .build();

            } catch (Exception e) {
                return ProviderHealth.unhealthy("Health check failed: " + e.getMessage());
            }
        });
    }

    @Override
    public Optional<ProviderMetrics> metrics() {
        return Optional.of(metrics);
    }

    @Override
    public void shutdown() {
        if (initialized.compareAndSet(true, false)) {
            sessionManager.shutdown();
            binding.backendFree();
            modelLastUsed.clear();
        }
    }

    private Set<DeviceType> buildSupportedDevices() {
        return config.gpuEnabled() ? Set.of(DeviceType.CPU, DeviceType.CUDA) : Set.of(DeviceType.CPU);
    }

    private void prewarmModels(java.util.List<String> modelIds) {
        modelIds.forEach(modelId -> {
            if (supports(modelId, TenantContext.of("system"))) {
                try {
                    sessionManager.getSession("system", modelId, config);
                } catch (Exception e) {
                    log.warn("Prewarm failed", e);
                }
            }
        });
    }

    private String resolveModelPath(String modelId) {
        return modelId.startsWith("/") ? modelId : config.modelBasePath() + "/" + modelId;
    }

    private InferenceRequest convertToInferenceRequest(ProviderRequest request) {
        String prompt = request.getMessages().stream()
                .map(Message::getContent)
                .collect(Collectors.joining("\n"));

        return InferenceRequest.builder()
                .model(request.getModel())
                .messages(request.getMessages())
                .parameter("prompt", prompt)
                .parameter("max_tokens", request.getMaxTokens())
                .parameter("temperature", request.getTemperature())
                .parameter("top_p", request.getTopP())
                .parameter("top_k", request.getParameter("top_k", Integer.class).orElse(config.defaultTopK()))
                .build();
    }

    private tech.kayys.golek.spi.context.RequestContext createRequestContext(
            TenantContext tenantContext,
            ProviderRequest request) {
        TenantContext effectiveContext = ensureTenantContext(tenantContext);
        return tech.kayys.golek.spi.context.RequestContext.create(
                effectiveContext.getTenantId().value(),
                "anonymous", // userId
                "session-" + java.util.UUID.randomUUID().toString());
    }

    private TenantContext ensureTenantContext(TenantContext tenantContext) {
        // Single-tenant default when multitenancy is disabled or context is absent.
        return tenantContext != null ? tenantContext : TenantContext.of("default");
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

    private void ensureInitialized() {
        if (!initialized.get())
            throw new IllegalStateException("GGUF Provider not initialized");
    }
}
