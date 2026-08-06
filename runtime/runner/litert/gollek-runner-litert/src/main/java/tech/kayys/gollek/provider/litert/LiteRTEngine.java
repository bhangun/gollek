package tech.kayys.gollek.provider.litert;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;
import org.jboss.logging.Logger;

import tech.kayys.gollek.spi.exception.ProviderException;
import tech.kayys.gollek.spi.inference.LocalInferenceEngine;
import tech.kayys.gollek.spi.inference.LocalInferenceEngine;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.exception.InferenceException;
import tech.kayys.alkhawarizm.core.tensor.DeviceType;
import tech.kayys.alkhawarizm.core.model.ModelFormat;
import tech.kayys.alkhawarizm.spi.model.ModalityType;
import tech.kayys.gollek.spi.observability.AdapterMetricsRecorder;
import tech.kayys.gollek.spi.observability.AdapterMetricSchema;
import tech.kayys.gollek.spi.observability.NoopAdapterMetricsRecorder;

import java.time.Duration;
import java.time.Instant;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * LiteRT provider for Gollek inference.
 *
 * <p>
 * Implements the LLMProvider SPI using LiteRT 2.0.
 * This class has been refactored to be POJO-first, allowing manual
 * instantiation
 * while remaining compatible with CDI-based environments.
 */
@jakarta.enterprise.context.ApplicationScoped
@io.quarkus.arc.Unremovable
public class LiteRTEngine implements LocalInferenceEngine {

    private static final Logger LOG = Logger.getLogger(LiteRTEngine.class);
    private static final String PROVIDER_ID = "litert";
    private static final String PROVIDER_VERSION = "2.0.0"; // Upgraded to 2.0 based on LiteRT 2.0

    @jakarta.inject.Inject
    LiteRTProviderConfig config;

    @jakarta.inject.Inject
    AdapterMetricsRecorder adapterMetricsRecorder = new NoopAdapterMetricsRecorder();

    @jakarta.inject.Inject
    LiteRTSessionManager sessionManager;

    private volatile boolean initialized = false;

    /**
     * Default constructor for SPI/CDI discovery.
     */
    public LiteRTEngine() {
    }

    /**
     * Manual constructor for standalone/JBang use cases.
     */
    public LiteRTEngine(LiteRTProviderConfig config) {
        this.config = config;
        this.sessionManager = new LiteRTSessionManager(config);
    }

    // Setters for manual/Quarkus-like injection if needed outside CDI
    public void setConfig(LiteRTProviderConfig config) {
        this.config = config;
    }

    public void setAdapterMetricsRecorder(AdapterMetricsRecorder recorder) {
        this.adapterMetricsRecorder = recorder != null ? recorder : new NoopAdapterMetricsRecorder();
    }

    public void setSessionManager(LiteRTSessionManager sessionManager) {
        this.sessionManager = sessionManager;
    }

    public void initialize() {
        if (config == null) {
            // Attempt to resolve config from cfg if possible, or expect it to be set
            // already
            LOG.warn("LiteRTProvider initialized without explicit config mapping.");
            config = defaultConfig();
        }

        if (sessionManager == null) {
            sessionManager = new LiteRTSessionManager(config);
        }

        if (sessionManager != null) {
            sessionManager.startEvictor();
        }

        initialized = true;
        LOG.infov("LiteRT 2.0 Provider initialized (GPU Auto-Metal: {0})",
                config != null && LiteRTDeviceSupport.shouldAutoMetal(config));
    }

    private LiteRTProviderConfig defaultConfig() {
        return new LiteRTProviderConfig() {
            @Override
            public boolean enabled() {
                return true;
            }

            @Override
            public String modelBasePath() {
                return Path.of(System.getProperty("user.home"), ".gollek", "models", "litert").toString();
            }

            @Override
            public int threads() {
                return Math.max(1, Runtime.getRuntime().availableProcessors() / 2);
            }

            @Override
            public boolean gpuEnabled() {
                return false;
            }

            @Override
            public boolean autoMetalEnabled() {
                return true;
            }

            @Override
            public boolean npuEnabled() {
                return false;
            }

            @Override
            public String gpuBackend() {
                return "auto";
            }

            @Override
            public String npuType() {
                return "auto";
            }

            @Override
            public Duration defaultTimeout() {
                return Duration.ofSeconds(30);
            }

            @Override
            public LiteRTProviderConfig.SessionConfig session() {
                return new LiteRTProviderConfig.SessionConfig() {
                    @Override
                    public int maxPerTenant() {
                        return 2;
                    }

                    @Override
                    public int idleTimeoutSeconds() {
                        return 300;
                    }

                    @Override
                    public int maxTotal() {
                        return 8;
                    }
                };
            }
        };
    }

    public boolean supports(String modelId, InferenceRequest request) {

        Path path = resolveModelPath(modelId, request);
        return path != null && Files.exists(path);
    }

    @Override
    public Uni<InferenceResponse> infer(InferenceRequest request) {
        return Uni.createFrom().item(() -> {
            ensureInitialized();

            String tenantId = resolveTenantId(request);
            String adapterId = PROVIDER_ID;

            adapterMetricsRecorder.recordSuccess(
                    AdapterMetricSchema.builder()
                            .adapterId(adapterId)
                            .modelId(request.getModel())
                            .operation("request_receive")

                            .build(),
                    0);

            Path modelPath = resolveModelPath(request.getModel(), request);
            boolean gpuEnabled = config != null && LiteRTDeviceSupport.effectiveGpuEnabled(config);
            String gpuBackend = config != null ? LiteRTDeviceSupport.resolveGpuBackend(config) : "auto";

            LiteRTRunnerConfig runnerConfig = new LiteRTRunnerConfig(
                    config != null ? config.threads() : 4,
                    gpuEnabled,
                    config != null && config.npuEnabled(),
                    gpuBackend,
                    config != null ? config.npuType() : "auto");

            LiteRTSessionManager.SessionContext sessionContext = null;
            Instant sessionAcquireStart = Instant.now();
            try {
                if (sessionManager == null) {
                    throw new InferenceException("LiteRT Session manager not initialized");
                }

                sessionContext = sessionManager.getSession(tenantId, request.getModel(), modelPath, runnerConfig);
                adapterMetricsRecorder.recordSuccess(
                        AdapterMetricSchema.builder()
                                .adapterId(adapterId)
                                .modelId(request.getModel())
                                .operation("session_acquire")

                                .build(),
                        Duration.between(sessionAcquireStart, Instant.now()).toMillis());

                InferenceRequest inferenceRequest = InferenceRequest.builder()
                        .requestId(request.getRequestId())
                        .model(request.getModel())
                        .messages(request.getMessages())
                        .parameters(request.getParameters())
                        .streaming(request.isStreaming())
                        .build();

                InferenceResponse response = sessionContext.runner().infer(inferenceRequest);

                return InferenceResponse.builder()
                        .requestId(response.getRequestId())
                        .content(response.getContent())
                        .model(request.getModel())
                        .durationMs(response.getDurationMs())
                        .tokensUsed(response.getTokensUsed())
                        .metadata(response.getMetadata())
                        .metadata("provider", PROVIDER_ID)
                        .metadata("litert_version", "2.0")
                        .build();
            } catch (Exception e) {
                String userFacingMessage = userFacingLiteRtFailureMessage(e);
                if (userFacingMessage != null) {
                    LOG.error("LiteRT 2.0 Inference failed: " + userFacingMessage);
                    throw new InferenceException(userFacingMessage);
                }
                LOG.error("LiteRT 2.0 Inference failed: " + e.getMessage(), e);
                throw new InferenceException("Inference failed", e);
            } finally {
                if (sessionContext != null && sessionManager != null) {
                    sessionManager.releaseSession(tenantId, request.getModel(), sessionContext);
                }
            }
        });
    }

    private String resolveTenantId(InferenceRequest request) {
        Object tenantId = request.getMetadata().get("tenantId");
        if (tenantId instanceof String && !((String) tenantId).isBlank()) {
            return (String) tenantId;
        }
        if (request.getUserId().isPresent()) {
            return request.getUserId().get();
        }
        if (request.getApiKey() != null && !request.getApiKey().isEmpty()) {
            return request.getApiKey();
        }
        return "community";
    }

    private Path resolveModelPath(String modelId, InferenceRequest request) {
        if (modelId == null)
            return null;

        LOG.debugf("Resolving model path for modelId: %s, request parameters: %s", modelId,
                (request != null ? request.getParameters() : "null"));

        // Check for explicit model_path parameter (e.g. from CLI --modelFile)
        if (request != null && request.getParameters().containsKey("model_path")) {
            Object pathObj = request.getParameters().get("model_path");
            LOG.debugf("Found model_path parameter: %s", pathObj);
            if (pathObj instanceof String s && !s.isBlank()) {
                try {
                    Path p = Paths.get(s);
                    if (Files.exists(p)) {
                        Path nativeGemmaLitertlm = findNativeGemmaLitertlm(p);
                        if (nativeGemmaLitertlm != null) {
                            LOG.debugf("Explicit model_path preferred native Gemma LiteRT-LM: %s", nativeGemmaLitertlm);
                            return nativeGemmaLitertlm;
                        }
                        Optional<Path> bestFile = LiteRTContainerParser.findBestModelFile(p);
                        LOG.debugf("Model path resolved to: %s", bestFile.orElse(p));
                        return bestFile.orElse(p);
                    } else {
                        LOG.debugf("Explicit model_path does not exist: %s", s);
                    }
                } catch (Exception e) {
                    LOG.debug("Failed to resolve explicit model_path: " + s);
                }
            }
        }

        try {
            Path targetPath = null;
            if (modelId.startsWith("file://")) {
                targetPath = Paths.get(modelId.substring("file://".length()));
            } else {
                Path asPath = Paths.get(modelId);
                if (asPath.isAbsolute() && Files.exists(asPath)) {
                    targetPath = asPath;
                } else if (config != null) {
                    Path basePath = Paths.get(config.modelBasePath());
                    Path modelDir = basePath.resolve(modelId);

                    if (Files.exists(modelDir)) {
                        targetPath = modelDir;
                    } else {
                        Path legacyPath = basePath.resolve(modelId + ".litertlm");
                        targetPath = Files.exists(legacyPath) ? legacyPath : modelDir;
                    }
                }
            }

            if (targetPath != null && Files.exists(targetPath)) {
                Path nativeGemmaLitertlm = findNativeGemmaLitertlm(targetPath);
                if (nativeGemmaLitertlm != null) {
                    LOG.debugf("Preferred native Gemma LiteRT-LM artifact: %s", nativeGemmaLitertlm);
                    return nativeGemmaLitertlm;
                }
                java.util.Optional<Path> bestFile = LiteRTContainerParser.findBestModelFile(targetPath);
                return bestFile.orElse(targetPath);
            }
            return targetPath;
        } catch (Exception e) {
            LOG.warn("Error resolving model path for " + modelId, e);
            return config != null ? Paths.get(config.modelBasePath()).resolve(modelId) : Paths.get(modelId);
        }
    }

    private Path findNativeGemmaLitertlm(Path sourcePath) {
        try {
            Path searchDir = Files.isDirectory(sourcePath) ? sourcePath : sourcePath.getParent();
            if (searchDir == null || !Files.isDirectory(searchDir)) {
                return null;
            }
            try (var stream = Files.list(searchDir)) {
                return stream
                        .filter(Files::isRegularFile)
                        .filter(path -> path.getFileName().toString().toLowerCase(Locale.ROOT).endsWith(".litertlm"))
                        .filter(path -> path.getFileName().toString().toLowerCase(Locale.ROOT).contains("gemma"))
                        .sorted((left, right) -> Integer.compare(
                                litertlmPreferenceScore(left),
                                litertlmPreferenceScore(right)))
                        .findFirst()
                        .orElse(null);
            }
        } catch (Exception e) {
            LOG.debugf("Failed to find native Gemma LiteRT-LM sibling for %s: %s", sourcePath, e.getMessage());
            return null;
        }
    }

    private int litertlmPreferenceScore(Path candidate) {
        String fileName = candidate.getFileName().toString().toLowerCase(Locale.ROOT);
        int score = 0;
        if (fileName.contains("qualcomm")) {
            score += 10;
        }
        if (!fileName.contains("gemma-4-e2b-it.litertlm")) {
            score += 1;
        }
        return score;
    }

    public tech.kayys.alkhawarizm.spi.model.HealthStatus health() {
        if (!initialized) {
            return tech.kayys.alkhawarizm.spi.model.HealthStatus.healthy("LiteRT Provider Operating Normally");
        }
        return tech.kayys.alkhawarizm.spi.model.HealthStatus.healthy("LiteRT Provider Operating Normally");
    }

    @Override
    public void shutdown() {
        if (sessionManager != null) {
            sessionManager.shutdown();
        }
        initialized = false;
        LOG.info("LiteRT 2.0 Provider shutdown complete");
    }

    @Override
    public io.smallrye.mutiny.Multi<StreamingInferenceChunk> stream(InferenceRequest request) {
        return Multi.createFrom().emitter(emitter -> {
            ensureInitialized();

            String tenantId = resolveTenantId(request);
            Path modelPath = resolveModelPath(request.getModel(), request);
            boolean gpuEnabled = config != null && LiteRTDeviceSupport.effectiveGpuEnabled(config);
            String gpuBackend = config != null ? LiteRTDeviceSupport.resolveGpuBackend(config) : "auto";

            LiteRTRunnerConfig runnerConfig = new LiteRTRunnerConfig(
                    config != null ? config.threads() : 4,
                    gpuEnabled,
                    config != null && config.npuEnabled(),
                    gpuBackend,
                    config != null ? config.npuType() : "auto");

            LiteRTSessionManager.SessionContext sessionContext = null;
            Instant started = Instant.now();
            AtomicInteger chunkIndex = new AtomicInteger(0);

            try {
                if (sessionManager == null) {
                    throw new InferenceException("LiteRT Session manager not initialized");
                }

                sessionContext = sessionManager.getSession(tenantId, request.getModel(), modelPath, runnerConfig);
                InferenceRequest inferenceRequest = InferenceRequest.builder()
                        .requestId(request.getRequestId())
                        .model(request.getModel())
                        .messages(request.getMessages())
                        .parameters(request.getParameters())
                        .streaming(true)
                        .build();

                sessionContext.runner().stream(inferenceRequest, delta -> {
                    Map<String, Object> metadata = Map.of(
                            "provider", PROVIDER_ID,
                            "litert_version", "2.0");
                    emitter.emit(new StreamingInferenceChunk(
                            request.getRequestId(),
                            chunkIndex.getAndIncrement(),
                            ModalityType.TEXT,
                            delta,
                            null,
                            false,
                            null,
                            null,
                            Instant.now(),
                            metadata,
                            null,
                            null,
                            null));
                });

                Map<String, Object> finalMetadata = new LinkedHashMap<>();
                finalMetadata.put("provider", PROVIDER_ID);
                finalMetadata.put("litert_version", "2.0");
                finalMetadata.put("duration_ms", Duration.between(started, Instant.now()).toMillis());
                finalMetadata.put("tokens.output", chunkIndex.get());
                emitter.emit(new StreamingInferenceChunk(
                        request.getRequestId(),
                        chunkIndex.getAndIncrement(),
                        ModalityType.TEXT,
                        null,
                        null,
                        true,
                        "stop",
                        null,
                        Instant.now(),
                        finalMetadata,
                        null,
                        null,
                        null));
                emitter.complete();
            } catch (Exception e) {
                String userFacingMessage = userFacingLiteRtFailureMessage(e);
                if (userFacingMessage != null) {
                    emitter.fail(new InferenceException(userFacingMessage));
                } else {
                    LOG.error("LiteRT 2.0 Streaming inference failed: " + e.getMessage(), e);
                    emitter.fail(new InferenceException("Streaming inference failed", e));
                }
            } finally {
                if (sessionContext != null && sessionManager != null) {
                    sessionManager.releaseSession(tenantId, request.getModel(), sessionContext);
                }
            }
        });
    }

    private static void put(Map<String, Object> target, String key, Object value) {
        if (value != null) {
            target.put(key, value);
        }
    }

    private static Long regularFileSize(Path path) {
        try {
            return path != null && Files.isRegularFile(path) ? Files.size(path) : null;
        } catch (Exception ignored) {
            return null;
        }
    }

    private static String litertArtifactType(Path path) {
        if (path == null || path.getFileName() == null) {
            return null;
        }
        String fileName = path.getFileName().toString().toLowerCase(Locale.ROOT);
        if (fileName.endsWith(".litertlm")) {
            return "litertlm";
        }
        if (fileName.endsWith(".tflite")) {
            return "tflite";
        }
        if (fileName.endsWith(".task")) {
            return "task";
        }
        if (Files.isDirectory(path)) {
            return "directory";
        }
        return "unknown";
    }

    private String userFacingLiteRtFailureMessage(Throwable throwable) {
        String rootMessage = rootCauseMessage(throwable);
        if (rootMessage == null || rootMessage.isBlank()) {
            return null;
        }
        if (rootMessage.contains("strict LiteRT mode")) {
            return rootMessage;
        }
        if (rootMessage.contains("Gemma 4 LiteRT .task runner is disabled by default")
                || rootMessage.contains("Raw Gemma LiteRT-LM signatures are disabled by default")) {
            return rootMessage;
        }
        return null;
    }

    private String rootCauseMessage(Throwable throwable) {
        if (throwable == null) {
            return null;
        }
        Throwable cursor = throwable;
        while (cursor.getCause() != null && cursor.getCause() != cursor) {
            cursor = cursor.getCause();
        }
        return cursor.getMessage();
    }

    private void ensureInitialized() {
        if (!initialized) {
            try {
                initialize();
            } catch (Exception e) {
                LOG.error("Lazy initialization failed", e);
            }
        }
    }
}
