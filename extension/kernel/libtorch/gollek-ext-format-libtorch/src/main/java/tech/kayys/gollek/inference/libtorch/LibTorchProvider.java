package tech.kayys.gollek.inference.libtorch;

import io.micrometer.core.instrument.MeterRegistry;
import io.opentelemetry.api.trace.Tracer;
import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.event.Observes;
import jakarta.inject.Inject;
import io.quarkus.runtime.StartupEvent;
import io.quarkus.runtime.ShutdownEvent;
import org.eclipse.microprofile.faulttolerance.CircuitBreaker;
import org.eclipse.microprofile.faulttolerance.Timeout;
import org.jboss.logging.Logger;

import tech.kayys.gollek.inference.libtorch.binding.LibTorchBinding;
import tech.kayys.gollek.inference.libtorch.binding.NativeLibraryLoader;
import tech.kayys.gollek.inference.libtorch.core.Tensor;
import tech.kayys.gollek.inference.libtorch.plugin.LibTorchPluginRegistry;
import tech.kayys.gollek.inference.libtorch.sampling.AutoregressiveGenerator;
import tech.kayys.gollek.inference.libtorch.sampling.SamplingStrategy;
import tech.kayys.gollek.inference.libtorch.sampling.SamplingStrategyFactory;
import tech.kayys.gollek.inference.libtorch.util.SafetensorsLoader;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.exception.ProviderException;
import tech.kayys.gollek.spi.model.DeviceType;
import tech.kayys.gollek.spi.model.ModelFormat;
import tech.kayys.gollek.spi.provider.*;
import tech.kayys.gollek.spi.stream.StreamChunk;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

/**
 * Golek LLM Provider for LibTorch/TorchScript models.
 * <p>
 * Implements the {@link StreamingProvider} SPI, enabling TorchScript model
 * inference
 * within the Golek engine. Supports three inference modes:
 * <ul>
 * <li><b>Sequential</b> (default): Each request is processed individually.</li>
 * <li><b>Batched</b> (opt-in): Requests are queued and processed in batches
 * for higher throughput via {@link ContinuousBatchingManager}.</li>
 * <li><b>Streaming</b>: Token-by-token generation via
 * {@link AutoregressiveGenerator}.</li>
 * </ul>
 */
@ApplicationScoped
public class LibTorchProvider implements StreamingProvider {

    private static final Logger log = Logger.getLogger(LibTorchProvider.class);
    private static final String PROVIDER_ID = "libtorch";
    private static final String PROVIDER_NAME = "LibTorch/TorchScript";
    private static final String PROVIDER_VERSION = "1.1.0";

    @Inject
    LibTorchProviderConfig config;

    @Inject
    LibTorchSessionManager sessionManager;

    @Inject
    LibTorchPluginRegistry pluginRegistry;

    @Inject
    ContinuousBatchingManager batchingManager;

    @Inject
    SafetensorsLoader safetensorsLoader;

    @Inject
    AutoregressiveGenerator generator;

    @Inject
    LibTorchChatTemplateService chatTemplateService;

    @Inject
    LibTorchMetrics metrics;

    @Inject
    Tracer tracer;

    @Inject
    MeterRegistry meterRegistry;

    @Inject
    LibTorchTokenizer tokenizer;

    private final AtomicReference<ProviderHealth.Status> status = new AtomicReference<>(ProviderHealth.Status.UNKNOWN);
    private final AtomicReference<String> startupFailure = new AtomicReference<>(null);
    private final AtomicLong requestCount = new AtomicLong(0);
    private final AtomicLong errorCount = new AtomicLong(0);
    private Instant startedAt;

    // ── Lifecycle ─────────────────────────────────────────────────────

    void onStart(@Observes StartupEvent ev) {
        if (!config.enabled()) {
            log.debug("LibTorch provider is disabled");
            status.set(ProviderHealth.Status.UNHEALTHY);
            return;
        }

        try {
            // Load native libraries
            var lookup = NativeLibraryLoader.load(config.nativeLib().libraryPath());
            LibTorchBinding binding = LibTorchBinding.initialize(lookup);

            if (!hasRequiredSymbols(binding)) {
                status.set(ProviderHealth.Status.UNHEALTHY);
                String message = "Missing required LibTorch wrapper symbols (at_jit_load/at_jit_module_forward). "
                        + "Install/load golek libtorch wrapper library.";
                startupFailure.set(message);
                log.warn(message);
                return;
            }

            // Initialize plugin registry
            pluginRegistry.initialize(binding);

            // Start session evictor for idle pooling
            sessionManager.startEvictor();

            status.set(ProviderHealth.Status.HEALTHY);
            startupFailure.set(null);
            startedAt = Instant.now();

            log.debugf("LibTorch provider started (plugins=%d, operations=%d, batching=%s)",
                    pluginRegistry.getPlugins().size(),
                    pluginRegistry.getAvailableOperations().size(),
                    config.batching().enabled() ? "enabled" : "disabled");

            // Warmup: preload models to eliminate cold-start latency
            warmupModels();
        } catch (Throwable e) {
            status.set(ProviderHealth.Status.UNHEALTHY);
            startupFailure.set(e.getMessage());
            log.errorf(e, "Failed to start LibTorch provider");
        }
    }

    void onShutdown(@Observes ShutdownEvent ev) {
        shutdown();
    }

    // ── LLMProvider SPI ───────────────────────────────────────────────

    @Override
    public String id() {
        return PROVIDER_ID;
    }

    @Override
    public String name() {
        return PROVIDER_NAME;
    }

    @Override
    public String version() {
        return PROVIDER_VERSION;
    }

    @Override
    public ProviderMetadata metadata() {
        return ProviderMetadata.builder()
                .providerId(PROVIDER_ID)
                .name(PROVIDER_NAME)
                .version(PROVIDER_VERSION)
                .description("LibTorch/TorchScript inference via JDK 25 FFM API with continuous batching")
                .vendor("Golek / Kayys")
                .homepage("https://pytorch.org/docs/stable/jit.html")
                .build();
    }

    @Override
    public ProviderCapabilities capabilities() {
        return ProviderCapabilities.builder()
                .streaming(true)
                .embeddings(true)
                .multimodal(false)
                .functionCalling(false)
                .toolCalling(false)
                .structuredOutputs(false)
                .maxContextTokens(0)
                .maxOutputTokens(0)
                .supportedFormats(Set.of(ModelFormat.TORCHSCRIPT, ModelFormat.PYTORCH, ModelFormat.SAFETENSORS))
                .supportedDevices(config.gpu().enabled()
                        ? Set.of(DeviceType.CPU, DeviceType.CUDA)
                        : Set.of(DeviceType.CPU))
                .features(buildFeatureSet())
                .build();
    }

    @Override
    public void initialize(ProviderConfig providerConfig)
            throws ProviderException.ProviderInitializationException {
        // Already initialized via CDI @Observes StartupEvent
        if (!NativeLibraryLoader.isLoaded()) {
            throw new ProviderException.ProviderInitializationException(
                    PROVIDER_ID, "Native LibTorch libraries not loaded");
        }
    }

    @Override
    public boolean supports(String modelId, ProviderRequest request) {
        if (!config.enabled() || status.get() != ProviderHealth.Status.HEALTHY) {
            return false;
        }

        Path modelPath = resolveModelPath(modelId);
        if (modelPath == null || !Files.exists(modelPath)) {
            return false;
        }

        String fileName = modelPath.getFileName().toString().toLowerCase();
        return configuredExtensions().stream().anyMatch(fileName::endsWith);
    }

    @Override
    public Uni<Boolean> isAvailable() {
        return Uni.createFrom().item(() -> status.get() == ProviderHealth.Status.HEALTHY);
    }

    @Override
    public Uni<ProviderHealth> health() {
        return Uni.createFrom().item(() -> {
            var currentStatus = status.get();
            var builder = ProviderHealth.builder()
                    .status(currentStatus)
                    .timestamp(Instant.now());

            if (currentStatus == ProviderHealth.Status.HEALTHY) {
                builder.message("LibTorch provider is healthy")
                        .detail("uptime", Duration.between(startedAt, Instant.now()).toString())
                        .detail("plugins", pluginRegistry.getPlugins().size())
                        .detail("operations", pluginRegistry.getAvailableOperations().size())
                        .detail("requests_total", requestCount.get())
                        .detail("errors_total", errorCount.get())
                        .detail("sessions_active", sessionManager.activeSessionCount())
                        .detail("sessions_idle", sessionManager.idleSessionCount())
                        .detail("sessions_total_created", sessionManager.totalCreatedCount())
                        .detail("batching_enabled", config.batching().enabled());

                if (config.batching().enabled()) {
                    builder.detail("batches_processed", batchingManager.getBatchCount())
                            .detail("batched_requests_total", batchingManager.getTotalBatchedRequests());
                }
            } else {
                builder.message("LibTorch provider is unhealthy");
                String reason = startupFailure.get();
                if (reason != null && !reason.isBlank()) {
                    builder.detail("reason", reason);
                }
            }

            return builder.build();
        });
    }

    @Override
    public void shutdown() {
        log.debug("Shutting down LibTorch provider...");
        sessionManager.shutdown();
        status.set(ProviderHealth.Status.UNKNOWN);
    }

    @Override
    @CircuitBreaker(requestVolumeThreshold = 10, failureRatio = 0.5, delay = 30000, successThreshold = 3)
    @Timeout(value = 30, unit = java.time.temporal.ChronoUnit.SECONDS)
    public Uni<InferenceResponse> infer(ProviderRequest request) {
        String tenantId = resolveTenantId(request);
        var span = tracer.spanBuilder("libtorch.infer")
                .setAttribute("model", request.getModel())
                .setAttribute("tenant", tenantId)
                .startSpan();

        metrics.recordRequest();
        Instant startTime = Instant.now();

        return Uni.createFrom().item(() -> {
            try {
                log.debugf("Starting inference for model=%s, tenant=%s", request.getModel(), tenantId);

                LibTorchGenerationParams params = convertToGenerationParams(request);
                String prompt = renderPrompt(request);

                if (prompt.isBlank()) {
                    return InferenceResponse.builder()
                            .requestId(request.getRequestId())
                            .model(request.getModel())
                            .content("")
                            .tokensUsed(0)
                            .build();
                }

                // Tokenize prompt using BPE tokenizer
                long[] promptTokens = tokenizer.encode(request.getModel(), prompt, true);

                // Use the AutoregressiveGenerator which manages its own session
                Path modelPath = sessionManager.resolveModelPath(request.getModel(), config);
                SamplingStrategy strategy = SamplingStrategyFactory.create(
                        "top_p", params.getTemperature(), params.getTopP(), params.getTopK());

                List<Long> generated = generator.generate(
                        tenantId, request.getModel(), modelPath,
                        promptTokens, strategy, params.getMaxTokens(), null);

                // Decode generated tokens back to text
                String responseText = tokenizer.decode(
                        request.getModel(), generated.stream().mapToLong(Long::longValue).toArray());

                InferenceResponse response = InferenceResponse.builder()
                        .requestId(request.getRequestId())
                        .model(request.getModel())
                        .content(responseText)
                        .tokensUsed(generated.size())
                        .finishReason(InferenceResponse.FinishReason.STOP)
                        .build();

                metrics.recordInference(request.getModel(), true, Duration.between(startTime, Instant.now()));
                metrics.recordTokensGenerated(generated.size());
                return response;

            } catch (Exception e) {
                metrics.recordInference(request.getModel(), false, Duration.ZERO);
                log.errorf(e, "Inference failed for model=%s", request.getModel());
                throw new ProviderException(PROVIDER_ID, "LibTorch inference failed: " + e.getMessage(), e,
                        isRetryable(e));
            } finally {
                span.end();
            }
        }).runSubscriptionOn(io.smallrye.mutiny.infrastructure.Infrastructure.getDefaultWorkerPool());
    }

    @Override
    public Multi<StreamChunk> inferStream(ProviderRequest request) {
        String tenantId = resolveTenantId(request);

        return Multi.createFrom().emitter(emitter -> {
            metrics.recordRequest();
            log.debugf("Starting streaming inference for model=%s, tenant=%s", request.getModel(), tenantId);

            try {
                LibTorchGenerationParams params = convertToGenerationParams(request);
                String prompt = renderPrompt(request);

                if (prompt.isBlank()) {
                    emitter.emit(StreamChunk.finalChunk(request.getRequestId(), 0, ""));
                    emitter.complete();
                    return;
                }

                long[] promptTokens = tokenizer.encode(request.getModel(), prompt, true);
                Path modelPath = sessionManager.resolveModelPath(request.getModel(), config);
                SamplingStrategy strategy = SamplingStrategyFactory.create(
                        "top_p", params.getTemperature(), params.getTopP(), params.getTopK());

                java.util.concurrent.atomic.AtomicInteger index = new java.util.concurrent.atomic.AtomicInteger(0);

                // Stream tokens via the generator's callback
                generator.generate(
                        tenantId, request.getModel(), modelPath,
                        promptTokens, strategy, params.getMaxTokens(),
                        token -> {
                            // Decode single token to text piece
                            String tokenText = tokenizer.decodeToken(request.getModel(), token);
                            emitter.emit(StreamChunk.of(
                                    request.getRequestId(),
                                    index.getAndIncrement(),
                                    tokenText));
                        });

                // Send final chunk
                emitter.emit(StreamChunk.finalChunk(request.getRequestId(), index.get(), ""));
                emitter.complete();

            } catch (Exception e) {
                metrics.recordFailure();
                log.errorf(e, "Streaming inference failed for model=%s", request.getModel());
                emitter.fail(new ProviderException(
                        PROVIDER_ID, "Streaming inference failed: " + e.getMessage(), e, isRetryable(e)));
            }
        });
    }

    private String renderPrompt(ProviderRequest request) {
        if (request.getMessages() != null && !request.getMessages().isEmpty()) {
            return chatTemplateService.renderChatML(request.getMessages());
        }
        return request.getParameter("prompt", String.class).orElse("");
    }

    private LibTorchGenerationParams convertToGenerationParams(ProviderRequest request) {
        return LibTorchGenerationParams.builder()
                .maxTokens(request.getParameter("max_tokens", Number.class)
                        .map(Number::intValue)
                        .orElse(config.generation().maxTokens()))
                .temperature(request.getParameter("temperature", Number.class)
                        .map(Number::floatValue)
                        .orElse(config.generation().temperature()))
                .topP(request.getParameter("top_p", Number.class)
                        .map(Number::floatValue)
                        .orElse(config.generation().topP()))
                .topK(request.getParameter("top_k", Number.class)
                        .map(Number::intValue)
                        .orElse(config.generation().topK()))
                .repeatPenalty(request.getParameter("repetition_penalty", Number.class)
                        .map(Number::floatValue)
                        .orElse(config.generation().repeatPenalty()))
                .repeatLastN(request.getParameter("repeat_last_n", Number.class)
                        .map(Number::intValue)
                        .orElse(config.generation().repeatLastN()))
                .build();
    }

    private boolean isRetryable(Throwable error) {
        if (error instanceof java.util.concurrent.TimeoutException)
            return true;
        if (error instanceof OutOfMemoryError)
            return false;
        String msg = error.getMessage();
        return msg == null || (!msg.toLowerCase().contains("not found") && !msg.toLowerCase().contains("invalid"));
    }

    private String resolveTenantId(ProviderRequest request) {
        if (request.getMetadata().containsKey("tenantId")) {
            return (String) request.getMetadata().get("tenantId");
        }
        return request.getUserId().orElse("default");
    }

    private Path resolveModelPath(String modelId) {
        if (modelId == null || modelId.isBlank()) {
            return null;
        }

        Path path = Path.of(modelId);
        if (path.isAbsolute()) {
            if (Files.exists(path)) {
                return path;
            }
            for (String ext : configuredExtensions()) {
                Path withExt = Path.of(modelId + ext);
                if (Files.exists(withExt)) {
                    return withExt;
                }
            }
            return path;
        }

        Path baseResolved = Path.of(config.model().basePath(), modelId);
        if (Files.exists(baseResolved)) {
            if (Files.isDirectory(baseResolved)) {
                Optional<Path> candidate = findBestModelFile(baseResolved);
                if (candidate.isPresent()) {
                    return candidate.get();
                }
            }
            return baseResolved;
        }
        for (String ext : configuredExtensions()) {
            Path withExt = Path.of(config.model().basePath(), modelId + ext);
            if (Files.exists(withExt)) {
                return withExt;
            }
        }

        String normalizedId = modelId.replace("/", "_");
        Path normalizedBase = Path.of(config.model().basePath(), normalizedId);
        if (Files.exists(normalizedBase)) {
            if (Files.isDirectory(normalizedBase)) {
                Optional<Path> candidate = findBestModelFile(normalizedBase);
                if (candidate.isPresent()) {
                    return candidate.get();
                }
            }
            return normalizedBase;
        }
        for (String ext : configuredExtensions()) {
            Path normalizedWithExt = Path.of(config.model().basePath(), normalizedId + ext);
            if (Files.exists(normalizedWithExt)) {
                return normalizedWithExt;
            }
        }

        // Fallback for origin checkpoint downloads stored under
        // ~/.gollek/models/djl/<repo-id>/...
        Path djlBase = Path.of(System.getProperty("user.home"), ".gollek", "models", "djl");
        Path djlResolved = djlBase.resolve(modelId);
        if (Files.exists(djlResolved)) {
            if (Files.isDirectory(djlResolved)) {
                Optional<Path> candidate = findBestModelFile(djlResolved);
                if (candidate.isPresent()) {
                    return candidate.get();
                }
            }
            return djlResolved;
        }

        return baseResolved;
    }

    private List<String> configuredExtensions() {
        String raw = config.model().extensions();
        if (raw == null || raw.isBlank()) {
            return List.of(".pt", ".pts", ".pth", ".bin", ".safetensors", ".safetensor");
        }
        List<String> exts = Arrays.stream(raw.split(","))
                .map(String::trim)
                .filter(s -> !s.isEmpty())
                .map(s -> s.startsWith(".") ? s.toLowerCase() : "." + s.toLowerCase())
                .toList();
        return exts.isEmpty() ? List.of(".pt", ".pts", ".pth", ".bin", ".safetensors", ".safetensor") : exts;
    }

    private Set<String> buildFeatureSet() {
        var features = new java.util.LinkedHashSet<>(List.of(
                "tensor-inference", "jit-scripting", "ffm-binding",
                "safetensors-loading", "streaming-generation",
                "sampling-strategies"));
        if (config.batching().enabled()) {
            features.add("continuous-batching");
        }
        return Set.copyOf(features);
    }

    /**
     * Load all weights from a safetensors file as a map of named Tensors.
     * <p>
     * Uses zero-copy memory mapping via the FFM API. Callers must close
     * the returned tensors when done.
     *
     * @param modelId model identifier (resolved to a .safetensors file)
     * @return map of tensor name → Tensor
     * @throws RuntimeException if loading fails
     */
    public Map<String, Tensor> loadSafetensorsWeights(String modelId) {
        Path path = resolveModelPath(modelId);
        if (path == null || !java.nio.file.Files.exists(path)) {
            throw new RuntimeException("Safetensors model not found: " + modelId);
        }
        try {
            return safetensorsLoader.loadAll(path);
        } catch (Exception e) {
            throw new RuntimeException("Failed to load safetensors weights: " + modelId, e);
        }
    }

    private void warmupModels() {
        if (!config.warmup().enabled()) {
            return;
        }
        config.warmup().models().ifPresent(modelList -> {
            String tenantId = config.warmup().tenantId();
            String[] modelIds = modelList.split(",");
            log.debugf("Warming up %d model(s)...", modelIds.length);

            for (String rawId : modelIds) {
                String modelId = rawId.trim();
                if (modelId.isEmpty())
                    continue;

                Path modelPath = resolveModelPath(modelId);
                if (modelPath == null || !Files.exists(modelPath)) {
                    log.warnf("Warmup: model not found, skipping: %s", modelId);
                    continue;
                }

                try {
                    LibTorchSessionManager.SessionContext session = sessionManager.acquire(tenantId, modelId,
                            modelPath);
                    try {
                        if (config.warmup().dummyForward()) {
                            // Run a dummy forward pass to trigger JIT compilation
                            // and CUDA kernel caching
                            try (Tensor dummy = Tensor.fromFloatArray(
                                    new float[] { 0.0f }, new long[] { 1, 1 });
                                    Tensor result = session.runner().forward(dummy)) {
                                log.debugf("Warmup: model '%s' loaded and JIT-compiled (output shape=%s)",
                                        modelId, Arrays.toString(result.shape()));
                            }
                        } else {
                            log.debugf("Warmup: model '%s' loaded (no dummy forward)", modelId);
                        }
                    } finally {
                        sessionManager.releaseSession(tenantId, modelId, session);
                    }
                } catch (Exception e) {
                    log.warnf(e, "Warmup: failed to preload model '%s'", modelId);
                    // Non-fatal — warmup failure should not prevent startup
                }
            }
            log.debug("Model warmup complete");
        });
    }

    private boolean hasRequiredSymbols(LibTorchBinding binding) {
        return binding.hasSymbol(LibTorchBinding.JIT_LOAD)
                && binding.hasSymbol(LibTorchBinding.JIT_MODULE_FORWARD)
                && binding.hasSymbol(LibTorchBinding.JIT_MODULE_FREE);
    }

    private Optional<Path> findBestModelFile(Path dir) {
        try (var paths = Files.walk(dir, 3)) {
            return paths
                    .filter(Files::isRegularFile)
                    .filter(p -> {
                        String name = p.getFileName().toString().toLowerCase(Locale.ROOT);
                        return configuredExtensions().stream().anyMatch(name::endsWith);
                    })
                    .sorted((a, b) -> Integer.compare(modelFilePriority(b), modelFilePriority(a)))
                    .findFirst();
        } catch (Exception ignored) {
            return Optional.empty();
        }
    }

    private int modelFilePriority(Path path) {
        String name = path.getFileName().toString().toLowerCase(Locale.ROOT);
        if (name.endsWith(".pt") || name.endsWith(".pts")) {
            return 50;
        }
        if (name.endsWith(".safetensors") || name.endsWith(".safetensor")) {
            return 40;
        }
        if (name.endsWith(".pth")) {
            return 30;
        }
        if (name.endsWith(".bin")) {
            return 20;
        }
        return 0;
    }
}
