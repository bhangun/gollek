package tech.kayys.gollek.provider.litert;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;
import org.jboss.logging.Logger;

import tech.kayys.gollek.spi.exception.ProviderException;
import tech.kayys.gollek.spi.provider.LLMProvider;
import tech.kayys.gollek.spi.provider.ProviderCapabilities;
import tech.kayys.gollek.spi.provider.ProviderConfig;
import tech.kayys.gollek.spi.provider.ProviderMetrics;
import tech.kayys.gollek.spi.provider.StreamingProvider;
import tech.kayys.gollek.spi.provider.ProviderHealth;
import tech.kayys.gollek.spi.provider.ProviderMetadata;
import tech.kayys.gollek.spi.provider.ProviderRequest;
import tech.kayys.gollek.spi.provider.AdapterCapabilityProfile;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.exception.InferenceException;
import tech.kayys.gollek.spi.model.DeviceType;
import tech.kayys.gollek.spi.model.ModelFormat;
import tech.kayys.gollek.spi.model.ModalityType;
import tech.kayys.gollek.spi.observability.AdapterMetricTagResolver;
import tech.kayys.gollek.spi.observability.AdapterMetricsRecorder;
import tech.kayys.gollek.spi.observability.AdapterMetricSchema;
import tech.kayys.gollek.spi.observability.NoopAdapterMetricsRecorder;

import java.time.Duration;
import java.time.Instant;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Optional;
import java.util.Set;

/**
 * LiteRT Provider for Wayang/Gollek.
 *
 * <p>Implements the LLMProvider SPI using LiteRT 2.0.
 * This class has been refactored to be POJO-first, allowing manual instantiation
 * while remaining compatible with CDI-based environments.
 */
@jakarta.enterprise.context.ApplicationScoped
@io.quarkus.arc.Unremovable
public class LiteRTProvider implements StreamingProvider {

    private static final Logger LOG = Logger.getLogger(LiteRTProvider.class);
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
    public LiteRTProvider() {
    }

    /**
     * Manual constructor for standalone/JBang use cases.
     */
    public LiteRTProvider(LiteRTProviderConfig config) {
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

    @Override
    public boolean isEnabled() {
        return config != null && config.enabled();
    }

    @Override
    public String id() {
        return PROVIDER_ID;
    }

    @Override
    public String name() {
        return "LiteRT 2.0 Provider (ARM64 Optimized)";
    }

    @Override
    public String version() {
        return PROVIDER_VERSION;
    }

    @Override
    public ProviderMetadata metadata() {
        return ProviderMetadata.builder()
                .providerId(PROVIDER_ID)
                .name(name())
                .version(PROVIDER_VERSION)
                .description("Stable LiteRT 2.0 inference provider for macOS/Linux ARM64")
                .vendor("Kayys Tech")
                .build();
    }

    @Override
    public ProviderCapabilities capabilities() {
        var features = new java.util.LinkedHashSet<>(Set.of("local_inference", "cpu_inference", "compiled_model_2.0"));
        features.addAll(AdapterCapabilityProfile.unsupportedWithMetrics().toFeatureFlags());
        return ProviderCapabilities.builder()
                .streaming(false)
                .functionCalling(false)
                .multimodal(false)
                .toolCalling(false)
                .embeddings(false)
                .maxContextTokens(2048)
                .maxOutputTokens(512)
                .supportedFormats(Set.of(ModelFormat.LITERT))
                .supportedDevices(config != null ? LiteRTDeviceSupport.supportedDevices(config) : Set.of(DeviceType.CPU))
                .features(Set.copyOf(features))
                .build();
    }

    @Override
    public void initialize(ProviderConfig cfg) throws ProviderException.ProviderInitializationException {
        if (config == null) {
            // Attempt to resolve config from cfg if possible, or expect it to be set already
            LOG.warn("LiteRTProvider initialized without explicit config mapping.");
        }
        
        if (sessionManager == null && config != null) {
            sessionManager = new LiteRTSessionManager(config);
        }

        if (sessionManager != null) {
            sessionManager.startEvictor();
        }
        
        initialized = true;
        LOG.infov("LiteRT 2.0 Provider initialized (GPU Auto-Metal: {0})", 
                config != null && LiteRTDeviceSupport.shouldAutoMetal(config));
    }

    @Override
    public boolean supports(String modelId, ProviderRequest request) {
        if (AdapterMetricTagResolver.hasAdapterRequest(request)) {
            return false;
        }
        Path path = resolveModelPath(modelId, request);
        return path != null && Files.exists(path);
    }

    @Override
    public Uni<InferenceResponse> infer(ProviderRequest request) {
        return Uni.createFrom().item(() -> {
            ensureInitialized();
            if (AdapterMetricTagResolver.hasAdapterRequest(request)) {
                throw new UnsupportedOperationException(
                        "LiteRT provider does not support adapters (adapter_unsupported).");
            }
            String tenantId = resolveTenantId(request);
            String adapterId = PROVIDER_ID;
            String adapterType = AdapterMetricTagResolver.resolveAdapterType(request);
            
            adapterMetricsRecorder.recordSuccess(
                    AdapterMetricSchema.builder()
                            .adapterId(adapterId)
                            .modelId(request.getModel())
                            .operation("request_receive")
                            .tags(java.util.Map.of("adapter_type", adapterType))
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
                                .tags(java.util.Map.of("adapter_type", adapterType))
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
                LOG.error("LiteRT 2.0 Inference failed: " + e.getMessage(), e);
                throw new InferenceException("Inference failed", e);
            } finally {
                if (sessionContext != null && sessionManager != null) {
                    sessionManager.releaseSession(tenantId, request.getModel(), sessionContext);
                }
            }
        });
    }

    private String resolveTenantId(ProviderRequest request) {
        Object tenantId = request.getMetadata().get("tenantId");
        if (tenantId instanceof String && !((String) tenantId).isBlank()) {
            return (String) tenantId;
        }
        if (request.getUserId().isPresent()) {
            return request.getUserId().get();
        }
        if (request.getApiKey().isPresent()) {
            return request.getApiKey().get();
        }
        return "community";
    }

    private Path resolveModelPath(String modelId, ProviderRequest request) {
        if (modelId == null) return null;
        
        LOG.debugf("Resolving model path for modelId: %s, request parameters: %s", modelId, (request != null ? request.getParameters() : "null"));

        // Check for explicit model_path parameter (e.g. from CLI --modelFile)
        if (request != null && request.getParameters().containsKey("model_path")) {
            Object pathObj = request.getParameters().get("model_path");
            LOG.debugf("Found model_path parameter: %s", pathObj);
            if (pathObj instanceof String s && !s.isBlank()) {
                try {
                    Path p = Paths.get(s);
                    if (Files.exists(p)) {
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
                java.util.Optional<Path> bestFile = LiteRTContainerParser.findBestModelFile(targetPath);
                return bestFile.orElse(targetPath);
            }
            return targetPath;
        } catch (Exception e) {
            LOG.warn("Error resolving model path for " + modelId, e);
            return config != null ? Paths.get(config.modelBasePath()).resolve(modelId) : Paths.get(modelId);
        }
    }

    @Override
    public Uni<ProviderHealth> health() {
        if (!initialized) {
            return Uni.createFrom().item(ProviderHealth.healthy("LiteRT provider pending initialization"));
        }
        return Uni.createFrom().item(ProviderHealth.healthy("LiteRT 2.0 provider operational"));
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
    public io.smallrye.mutiny.Multi<StreamingInferenceChunk> inferStream(ProviderRequest request) {
        return infer(request).onItem().transformToMulti(response -> {
            return io.smallrye.mutiny.Multi.createFrom().item(
                new StreamingInferenceChunk(
                    response.getRequestId(),
                    0,
                    ModalityType.TEXT,
                    response.getContent(),
                    null,
                    true,
                    "stop",
                    null,
                    java.time.Instant.now(),
                    response.getMetadata()
                )
            );
        });
    }

    private void ensureInitialized() {
        if (!initialized) {
            try {
                initialize(null);
            } catch (Exception e) {
                LOG.error("Lazy initialization failed", e);
            }
        }
    }
}
