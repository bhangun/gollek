package tech.kayys.gollek.provider.litert;

import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;

import tech.kayys.gollek.spi.exception.ProviderException;
import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.model.DeviceType;
import tech.kayys.gollek.spi.model.ModelFormat;
import tech.kayys.gollek.spi.provider.*;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

@ApplicationScoped
public class LiteRTProvider implements LLMProvider {

    private static final Logger LOG = Logger.getLogger(LiteRTProvider.class);
    private static final String PROVIDER_ID = "litert";
    private static final String PROVIDER_VERSION = "1.0.0";

    @Inject
    LiteRTProviderConfig config;

    private final Map<String, LiteRTCpuRunner> runners = new ConcurrentHashMap<>();
    private volatile boolean initialized = false;

    @Override
    public String id() {
        return PROVIDER_ID;
    }

    @Override
    public String name() {
        return "LiteRT Provider (TFLite)";
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
                .description("Local LiteRT (TFLite) inference provider")
                .vendor("Kayys Tech")
                .build();
    }

    @Override
    public ProviderCapabilities capabilities() {
        return ProviderCapabilities.builder()
                .streaming(false)
                .functionCalling(false)
                .multimodal(false)
                .toolCalling(false)
                .embeddings(false)
                .maxContextTokens(2048)
                .maxOutputTokens(512)
                .supportedFormats(Set.of(ModelFormat.LITERT))
                .supportedDevices(Set.of(DeviceType.CPU))
                .features(Set.of("local_inference", "cpu_inference"))
                .build();
    }

    @Override
    public void initialize(ProviderConfig cfg) throws ProviderException.ProviderInitializationException {
        initialized = true;
    }

    @Override
    public boolean supports(String modelId, ProviderRequest request) {
        Path path = resolveModelPath(modelId);
        return Files.exists(path);
    }

    @Override
    public Uni<InferenceResponse> infer(ProviderRequest request) {
        return Uni.createFrom().item(() -> {
            ensureInitialized();
            String tenantId = resolveTenantId(request);

            LiteRTCpuRunner runner = getOrCreateRunner(
                    tenantId,
                    request.getModel());

            InferenceRequest inferenceRequest = InferenceRequest.builder()
                    .requestId(request.getRequestId())
                    .model(request.getModel())
                    .messages(request.getMessages())
                    .parameters(request.getParameters())
                    .streaming(request.isStreaming())
                    .build();

            InferenceResponse response = runner.infer(inferenceRequest);

            return InferenceResponse.builder()
                    .requestId(response.getRequestId())
                    .content(response.getContent())
                    .model(request.getModel())
                    .durationMs(response.getDurationMs())
                    .tokensUsed(response.getTokensUsed())
                    .metadata(response.getMetadata())
                    .metadata("provider", PROVIDER_ID)
                    .build();
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
        // Try apiKey
        if (request.getApiKey().isPresent()) {
            return request.getApiKey().get();
        }
        return "community";
    }

    private LiteRTCpuRunner getOrCreateRunner(String requestId, String modelId) {
        String key = requestId + ":" + modelId;
        return runners.computeIfAbsent(key, k -> {
            Path modelPath = resolveModelPath(modelId);
            LiteRTCpuRunner runner = new LiteRTCpuRunner();
            LiteRTRunnerConfig runnerConfig = new LiteRTRunnerConfig(
                    config.threads(),
                    config.gpuEnabled(),
                    config.npuEnabled(),
                    config.gpuBackend(),
                    config.npuType());
            runner.initialize(modelPath, runnerConfig);
            LOG.infof("LiteRT runner initialized for %s", modelId);
            return runner;
        });
    }

    private Path resolveModelPath(String modelId) {
        if (modelId.startsWith("file://")) {
            return Paths.get(modelId.substring("file://".length()));
        }

        Path asPath = Paths.get(modelId);
        if (asPath.isAbsolute() && Files.exists(asPath)) {
            return asPath;
        }

        Path basePath = Paths.get(config.modelBasePath());
        if (modelId.endsWith(".tflite")) {
            return basePath.resolve(modelId);
        }

        return basePath.resolve(modelId + ".tflite");
    }

    @Override
    public Uni<ProviderHealth> health() {
        if (!initialized) {
            return Uni.createFrom().item(ProviderHealth.unhealthy("Provider not initialized"));
        }
        return Uni.createFrom().item(ProviderHealth.healthy("LiteRT provider operational"));
    }

    @Override
    public void shutdown() {
        runners.values().forEach(LiteRTCpuRunner::close);
        runners.clear();
        initialized = false;
    }

    private void ensureInitialized() {
        if (!initialized) {
            initialized = true;
            LOG.info("LiteRT provider lazily initialized");
        }
    }
}
