package tech.kayys.gollek.inference.libtorch;

import tech.kayys.gollek.tokenizer.spi.Tokenizer;
import tech.kayys.gollek.tokenizer.spi.EncodeOptions;
import tech.kayys.gollek.tokenizer.spi.DecodeOptions;
import tech.kayys.gollek.tokenizer.spi.StreamingDecoder;

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
import tech.kayys.gollek.inference.libtorch.config.LibTorchProviderConfig;
import tech.kayys.gollek.inference.libtorch.core.TorchTensor;
import tech.kayys.gollek.inference.libtorch.plugin.LibTorchPluginRegistry;
import tech.kayys.gollek.runtime.inference.batch.ContinuousBatchScheduler;
import tech.kayys.gollek.inference.libtorch.sampling.AutoregressiveGenerator;
import tech.kayys.gollek.inference.libtorch.sampling.SamplingStrategy;
import tech.kayys.gollek.inference.libtorch.sampling.SamplingStrategyFactory;
import tech.kayys.gollek.inference.libtorch.util.SafetensorsLoader;
import tech.kayys.gollek.spi.observability.AdapterSpec;
import tech.kayys.gollek.spi.observability.AdapterSpecResolver;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.model.ModelFormat;
import tech.kayys.gollek.spi.provider.AdapterCapabilityProfile;
import tech.kayys.gollek.spi.provider.ProviderCapabilities;
import tech.kayys.gollek.spi.provider.ProviderConfig;
import tech.kayys.gollek.spi.provider.ProviderHealth;
import tech.kayys.gollek.spi.provider.ProviderMetadata;
import tech.kayys.gollek.spi.provider.ProviderRequest;
import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;
import tech.kayys.gollek.spi.provider.StreamingProvider;
import tech.kayys.gollek.spi.exception.ProviderException;

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
 * for higher throughput via {@link ContinuousBatchScheduler}.</li>
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

    ContinuousBatchScheduler batchScheduler;

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
    LibTorchTokenizerManager tokenizerManager;

    private final AtomicReference<ProviderHealth.Status> status = new AtomicReference<>(ProviderHealth.Status.UNKNOWN);
    private final AtomicReference<String> startupFailure = new AtomicReference<>(null);
    private final AtomicReference<LibTorchAdvancedModeResolver.EffectiveAdvancedMode> effectiveAdvancedMode = new AtomicReference<>(
            LibTorchAdvancedModeResolver.EffectiveAdvancedMode.baseline(
                    "startup.pending",
                    Set.of()));
    private final AtomicReference<LibTorchExecutionHints> lastExecutionHints = new AtomicReference<>(
            LibTorchExecutionHints.baseline());
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
            batchScheduler = new ContinuousBatchScheduler(config.batching().maxBatchSize());

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

            resolveAndLogAdvancedMode(binding);

            // Start session evictor for idle pooling
            sessionManager.startEvictor();

            if (LibTorchDeviceSupport.shouldAutoMps(config)) {
                log.info("LibTorch auto-MPS enabled (Apple Silicon)");
            }

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

    @jakarta.annotation.PreDestroy
    public void shutdown() {
        shutdown();
    }

    // ── LLMProvider SPI ───────────────────────────────────────────────

    @Override
    public boolean isEnabled() {
        return config.enabled();
    }

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
                .homepage("https://pylibtorch.org/docs/stable/jit.html")
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
                .supportedDevices(LibTorchDeviceSupport.supportedDevices(config))
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
                        .detail("batching_enabled", config.batching().enabled())
                        .detail("advanced_effective_enabled", effectiveAdvancedMode.get().advancedEnabled())
                        .detail("advanced_attention_mode", effectiveAdvancedMode.get().attentionMode())
                        .detail("advanced_reason", effectiveAdvancedMode.get().reason())
                        .detail("advanced_allowed_gpu_sm", effectiveAdvancedMode.get().allowedGpuSm().toString())
                        .detail("advanced_detected_gpu_sm",
                                effectiveAdvancedMode.get().detectedGpuSm().map(Object::toString).orElse("unknown"))
                        .detail("advanced_fp8_rowwise_enabled", effectiveAdvancedMode.get().fp8RowwiseEnabled())
                        .detail("advanced_fp8_rowwise_requested", effectiveAdvancedMode.get().fp8RowwiseEnabled())
                        .detail("advanced_sage_attention2_requested",
                                effectiveAdvancedMode.get().sageAttention2Requested())
                        .detail("advanced_sage_attention2_enabled", effectiveAdvancedMode.get().sageAttention2Enabled())
                        .detail("advanced_sage_attention2_rollback_reason",
                                effectiveAdvancedMode.get().sageAttention2RollbackReason())
                        .detail("advanced_sage_attention2_active", lastExecutionHints.get().sageAttention2Enabled())
                        .detail("advanced_sage_attention2_reason", lastExecutionHints.get().sageAttention2Reason())
                        .detail("advanced_fp8_rowwise_active", lastExecutionHints.get().fp8RowwiseEnabled())
                        .detail("advanced_fp8_rowwise_reason", lastExecutionHints.get().fp8RowwiseReason())
                        .detail("advanced_fp8_rowwise_scale_count", lastExecutionHints.get().fp8RowwiseScaleCount())
                        .detail("advanced_fp8_rowwise_scale_mean", lastExecutionHints.get().fp8RowwiseScaleMean())
                        .detail("advanced_fp8_rowwise_calibration_source",
                                lastExecutionHints.get().fp8RowwiseCalibrationSource());

                if (config.batching().enabled()) {
                    builder.detail("active_requests", batchScheduler.activeCount())
                            .detail("pending_requests", batchScheduler.pendingCount());
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
        var advanced = effectiveAdvancedMode.get();
        var span = tracer.spanBuilder("libtorch.infer")
                .setAttribute("model", request.getModel())
                .setAttribute("tenant", tenantId)
                .setAttribute("advanced.enabled", advanced.advancedEnabled())
                .setAttribute("advanced.attention_mode", advanced.attentionMode())
                .setAttribute("advanced.reason", advanced.reason())
                .startSpan();

        metrics.recordRequest();
        Instant startTime = Instant.now();

        return Uni.createFrom().item(() -> {
            try {
                log.debugf("Starting inference for model=%s, tenant=%s", request.getModel(), tenantId);
                AdapterSpec adapterSpec = resolveAdapterSpec(request);

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

                // Tokenize prompt using the unified Tokenizer
                Tokenizer tokenizer = tokenizerManager.getTokenizer(request.getModel());
                long[] promptTokens = tokenizer.encode(prompt, EncodeOptions.defaultOptions());

                // Use the AutoregressiveGenerator which manages its own session
                Path modelPath = sessionManager.resolveModelPath(tenantId, request.getModel(), config, adapterSpec);
                SamplingStrategy strategy = SamplingStrategyFactory.create(
                        "top_p", params.getTemperature(), params.getTopP(), params.getTopK());
                LibTorchExecutionHints executionHints = resolveExecutionHints(modelPath, tenantId, request.getModel());
                span.setAttribute("advanced.sage_attention2_requested", executionHints.sageAttention2Requested());
                span.setAttribute("advanced.sage_attention2_active", executionHints.sageAttention2Enabled());
                span.setAttribute("advanced.sage_attention2_reason", executionHints.sageAttention2Reason());
                span.setAttribute("advanced.fp8_rowwise_active", executionHints.fp8RowwiseEnabled());
                span.setAttribute("advanced.fp8_rowwise_reason", executionHints.fp8RowwiseReason());
                span.setAttribute("advanced.fp8_rowwise_scale_count", executionHints.fp8RowwiseScaleCount());
                span.setAttribute("advanced.fp8_rowwise_scale_mean", executionHints.fp8RowwiseScaleMean());

                List<Long> generated = generator.generate(
                        tenantId, request.getModel(), modelPath,
                        promptTokens, strategy, params.getMaxTokens(), null, adapterSpec, executionHints);

                // Decode generated tokens back to text
                String responseText = tokenizer.decode(
                        generated.stream().mapToLong(Long::longValue).toArray(),
                        DecodeOptions.defaultOptions());

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
                        tech.kayys.gollek.error.ErrorCode.RUNTIME_INFERENCE_FAILED, isRetryable(e));
            } finally {
                span.end();
            }
        }).runSubscriptionOn(io.smallrye.mutiny.infrastructure.Infrastructure.getDefaultWorkerPool());
    }

    @Override
    public Multi<StreamingInferenceChunk> inferStream(ProviderRequest request) {
        String tenantId = resolveTenantId(request);

        return Multi.createFrom().emitter(emitter -> {
            metrics.recordRequest();
            log.debugf("Starting streaming inference for model=%s, tenant=%s", request.getModel(), tenantId);

            try {
                AdapterSpec adapterSpec = resolveAdapterSpec(request);
                LibTorchGenerationParams params = convertToGenerationParams(request);
                String prompt = renderPrompt(request);

                if (prompt.isBlank()) {
                    emitter.emit(StreamingInferenceChunk.finalChunk(request.getRequestId(), 0, ""));
                    emitter.complete();
                    return;
                }

                Tokenizer tokenizer = tokenizerManager.getTokenizer(request.getModel());
                long[] promptTokens = tokenizer.encode(prompt, EncodeOptions.defaultOptions());
                Path modelPath = sessionManager.resolveModelPath(tenantId, request.getModel(), config, adapterSpec);
                SamplingStrategy strategy = SamplingStrategyFactory.create(
                        "top_p", params.getTemperature(), params.getTopP(), params.getTopK());
                LibTorchExecutionHints executionHints = resolveExecutionHints(modelPath, tenantId, request.getModel());

                java.util.concurrent.atomic.AtomicInteger index = new java.util.concurrent.atomic.AtomicInteger(0);

                StreamingDecoder decoder = new StreamingDecoder(tokenizer, DecodeOptions.defaultOptions());

                // Stream tokens via the generator's callback
                generator.generate(
                        tenantId, request.getModel(), modelPath,
                        promptTokens, strategy, params.getMaxTokens(),
                        token -> {
                            // Robust decoding via StreamingDecoder
                            String tokenText = decoder.decodeNext(token);
                            emitter.emit(StreamingInferenceChunk.of(
                                    request.getRequestId(),
                                    index.getAndIncrement(),
                                    tokenText));
                        }, adapterSpec, executionHints);

                // Send final chunk
                emitter.emit(StreamingInferenceChunk.finalChunk(request.getRequestId(), index.get(), ""));
                emitter.complete();

            } catch (Exception e) {
                metrics.recordFailure();
                log.errorf(e, "Streaming inference failed for model=%s", request.getModel());
                emitter.fail(new ProviderException(
                        PROVIDER_ID, "Streaming inference failed: " + e.getMessage(), e,
                        tech.kayys.gollek.error.ErrorCode.RUNTIME_INFERENCE_FAILED, isRetryable(e)));
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
        var advanced = effectiveAdvancedMode.get();
        if (advanced.advancedEnabled()) {
            features.add("advanced_cuda_path");
            features.add("attention_mode_" + advanced.attentionMode());
            if (advanced.fp8RowwiseEnabled()) {
                features.add("fp8_rowwise_weights");
            }
            if (advanced.sageAttention2Enabled()) {
                features.add("sage_attention2_experimental");
            }
        } else {
            features.add("attention_mode_baseline");
        }
        features.addAll(AdapterCapabilityProfile.builder()
                .adapterSupported(config.adapter() != null && config.adapter().enabled())
                .adapterTypes(Set.of("lora"))
                .runtimeApply(LibTorchBinding.isInitialized()
                        && LibTorchBinding.getInstance().hasSymbol(LibTorchBinding.JIT_APPLY_LORA))
                .precompiledModelPath(config.adapter() != null && config.adapter().allowPrecompiledModelPath())
                .rolloutGuard(config.adapter() != null && config.adapter().rolloutGuardEnabled())
                .metricsSchema(true)
                .build()
                .toFeatureFlags());
        if (LibTorchBinding.isInitialized()
                && LibTorchBinding.getInstance().hasSymbol(LibTorchBinding.JIT_APPLY_LORA)) {
            features.add("adapter_runtime_lora_patch");
        }
        if (config.batching().enabled()) {
            features.add("continuous-batching");
        }
        return Set.copyOf(features);
    }

    private void resolveAndLogAdvancedMode(LibTorchBinding binding) {
        LibTorchAdvancedModeResolver resolver = new LibTorchAdvancedModeResolver();
        LibTorchAdvancedModeResolver.EffectiveAdvancedMode resolved = resolver.resolve(config, binding);
        effectiveAdvancedMode.set(resolved);
        metrics.recordAdvancedModeResolution(resolved);
        if (resolved.advancedEnabled()) {
            log.infof(
                    "LibTorch advanced path enabled: attention=%s, fp8Rowwise=%s, sageAttention2=%s, sm=%s, allowedSm=%s",
                    resolved.attentionMode(),
                    resolved.fp8RowwiseEnabled(),
                    resolved.sageAttention2Enabled(),
                    resolved.detectedGpuSm().map(Object::toString).orElse("unknown"),
                    resolved.allowedGpuSm());
            if (resolved.sageAttention2Requested() && !resolved.sageAttention2Enabled()) {
                log.infof("SageAttention2 requested but rolled back (reason=%s)",
                        resolved.sageAttention2RollbackReason());
            }
        } else {
            log.infof("LibTorch advanced path fallback to baseline (reason=%s, detectedSm=%s, allowedSm=%s)",
                    resolved.reason(),
                    resolved.detectedGpuSm().map(Object::toString).orElse("unknown"),
                    resolved.allowedGpuSm());
        }
    }

    private LibTorchExecutionHints resolveExecutionHints(Path modelPath, String tenantId, String modelId) {
        var advanced = effectiveAdvancedMode.get();
        SageAttention2Decision sageDecision = resolveSageAttention2Decision(advanced, tenantId, modelId);
        Fp8RowwiseDecision rowwiseDecision = resolveFp8RowwiseDecision(advanced, tenantId, modelId);
        if (advanced != null
                && advanced.advancedEnabled()
                && "hybrid_fp8_bf16".equals(advanced.attentionMode())) {
            LibTorchFp8RowwisePlanner.RowwisePlan rowwisePlan;
            if (rowwiseDecision.enabled()) {
                LibTorchAdvancedModeResolver.EffectiveAdvancedMode rowwiseMode = new LibTorchAdvancedModeResolver.EffectiveAdvancedMode(
                        advanced.advancedEnabled(),
                        advanced.attentionMode(),
                        true,
                        advanced.sageAttention2Requested(),
                        advanced.sageAttention2Enabled(),
                        advanced.sageAttention2RollbackReason(),
                        advanced.reason(),
                        advanced.detectedGpuSm(),
                        advanced.allowedGpuSm());
                rowwisePlan = new LibTorchFp8RowwisePlanner().resolve(modelPath, rowwiseMode);
            } else {
                rowwisePlan = new LibTorchFp8RowwisePlanner.RowwisePlan(
                        false,
                        rowwiseDecision.reason(),
                        0,
                        0.0d,
                        "none",
                        null);
            }
            if (!rowwisePlan.enabled() && rowwiseDecision.requested()) {
                log.infof("FP8 rowwise disabled for current run (reason=%s, model=%s)", rowwisePlan.reason(),
                        modelPath);
            }
            LibTorchExecutionHints hints = LibTorchExecutionHints.hybridFp8Bf16(
                    sageDecision.requested(),
                    sageDecision.enabled(),
                    sageDecision.reason(),
                    rowwisePlan.enabled(),
                    rowwisePlan.reason(),
                    rowwisePlan.scaleCount(),
                    rowwisePlan.scaleMean(),
                    rowwisePlan.calibrationSource(),
                    rowwisePlan.rowScales());
            lastExecutionHints.set(hints);
            if (metrics != null) {
                metrics.recordFp8RowwiseDecision(hints.fp8RowwiseEnabled(), hints.fp8RowwiseReason());
            }
            return hints;
        }
        LibTorchExecutionHints baseline = LibTorchExecutionHints.baselineWithSageState(
                sageDecision.requested(),
                sageDecision.enabled(),
                sageDecision.reason());
        lastExecutionHints.set(baseline);
        if (metrics != null) {
            metrics.recordFp8RowwiseDecision(false, baseline.fp8RowwiseReason());
        }
        return baseline;
    }

    private SageAttention2Decision resolveSageAttention2Decision(
            LibTorchAdvancedModeResolver.EffectiveAdvancedMode advanced,
            String tenantId,
            String modelId) {
        if (advanced == null || !advanced.sageAttention2Requested()) {
            return SageAttention2Decision.none();
        }
        if (isBlocked(config.advanced().sageAttention2BlockedTenants(), tenantId)) {
            return new SageAttention2Decision(true, false, "sageattention2.canary.denied.tenant");
        }
        if (isBlocked(config.advanced().sageAttention2BlockedModels(), modelId)) {
            return new SageAttention2Decision(true, false, "sageattention2.canary.denied.model");
        }
        if (!isAllowed(config.advanced().sageAttention2AllowedTenants(), tenantId)) {
            return new SageAttention2Decision(true, false, "sageattention2.canary.blocked.tenant");
        }
        if (!isAllowed(config.advanced().sageAttention2AllowedModels(), modelId)) {
            return new SageAttention2Decision(true, false, "sageattention2.canary.blocked.model");
        }
        if (advanced.sageAttention2Enabled()) {
            return new SageAttention2Decision(true, true, "sageattention2.enabled");
        }
        String reason = advanced.sageAttention2RollbackReason();
        return new SageAttention2Decision(true, false, reason == null || reason.isBlank() ? "unknown" : reason);
    }

    private Fp8RowwiseDecision resolveFp8RowwiseDecision(
            LibTorchAdvancedModeResolver.EffectiveAdvancedMode advanced,
            String tenantId,
            String modelId) {
        if (advanced == null || !advanced.fp8RowwiseEnabled()) {
            return Fp8RowwiseDecision.none();
        }
        if (isBlocked(config.advanced().fp8RowwiseBlockedTenants(), tenantId)) {
            return new Fp8RowwiseDecision(true, false, "fp8.rowwise.canary.denied.tenant");
        }
        if (isBlocked(config.advanced().fp8RowwiseBlockedModels(), modelId)) {
            return new Fp8RowwiseDecision(true, false, "fp8.rowwise.canary.denied.model");
        }
        if (!isAllowed(config.advanced().fp8RowwiseAllowedTenants(), tenantId)) {
            return new Fp8RowwiseDecision(true, false, "fp8.rowwise.canary.blocked.tenant");
        }
        if (!isAllowed(config.advanced().fp8RowwiseAllowedModels(), modelId)) {
            return new Fp8RowwiseDecision(true, false, "fp8.rowwise.canary.blocked.model");
        }
        return new Fp8RowwiseDecision(true, true, "fp8.rowwise.enabled");
    }

    private boolean isAllowed(Optional<List<String>> allowList, String value) {
        if (allowList == null || allowList.isEmpty() || allowList.get().isEmpty()) {
            return true;
        }
        if (value == null || value.isBlank()) {
            return false;
        }
        return allowList.get().stream()
                .filter(s -> s != null && !s.isBlank())
                .anyMatch(s -> s.trim().equalsIgnoreCase(value));
    }

    private boolean isBlocked(Optional<List<String>> denyList, String value) {
        if (denyList == null || denyList.isEmpty() || denyList.get().isEmpty()) {
            return false;
        }
        if (value == null || value.isBlank()) {
            return false;
        }
        return denyList.get().stream()
                .filter(s -> s != null && !s.isBlank())
                .anyMatch(s -> s.trim().equalsIgnoreCase(value));
    }

    private record SageAttention2Decision(boolean requested, boolean enabled, String reason) {
        static SageAttention2Decision none() {
            return new SageAttention2Decision(false, false, "none");
        }
    }

    private record Fp8RowwiseDecision(boolean requested, boolean enabled, String reason) {
        static Fp8RowwiseDecision none() {
            return new Fp8RowwiseDecision(false, false, "fp8.rowwise.disabled");
        }
    }

    private AdapterSpec resolveAdapterSpec(ProviderRequest request) {
        if (config.adapter() == null || !config.adapter().enabled()) {
            return null;
        }
        return AdapterSpecResolver.fromProviderRequest(request, 1.0f).orElse(null);
    }

    /**
     * Load all weights from a safetensors file as a map of named Tensors.
     * <p>
     * Uses zero-copy memory mapping via the FFM API. Callers must close
     * the returned tensors when done.
     *
     * @param modelId model identifier (resolved to a .safetensors file)
     * @return map of tensor name → TorchTensor
     * @throws RuntimeException if loading fails
     */
    public Map<String, TorchTensor> loadSafetensorsWeights(String modelId) {
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
                    LibTorchSessionManager.SessionContext session = sessionManager.getSession(tenantId, modelId, config);
                    try {
                        if (config.warmup().dummyForward()) {
                            // Run a dummy forward pass to trigger JIT compilation
                            // and CUDA kernel caching
                            try (TorchTensor dummy = TorchTensor.fromFloatArray(
                                    new float[] { 0.0f }, new long[] { 1, 1 });
                                    TorchTensor result = session.runner().forward(dummy)) {
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
