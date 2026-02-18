package tech.kayys.gollek.engine.provider.adapter;

import io.smallrye.mutiny.Uni;
import io.smallrye.mutiny.Multi;

import tech.kayys.gollek.spi.inference.InferenceRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.model.ModelManifest;
import tech.kayys.gollek.spi.provider.ProviderCapabilities;
import tech.kayys.gollek.spi.provider.ProviderConfig;
import tech.kayys.gollek.spi.provider.ProviderHealth;
import tech.kayys.gollek.spi.provider.ProviderMetadata;
import tech.kayys.gollek.spi.provider.ProviderRequest;
import tech.kayys.gollek.spi.provider.StreamingProvider;
import tech.kayys.gollek.engine.model.ModelRunner;
import tech.kayys.gollek.engine.model.ModelRunnerFactory;
import tech.kayys.gollek.model.exception.ModelNotFoundException;
import tech.kayys.gollek.engine.model.CachedModelRepository;

import tech.kayys.gollek.spi.stream.StreamChunk;

import java.time.Duration;

/**
 * Adapter that exposes a local ModelRunner as an LLMProvider.
 * This enables the unified routing and execution of both local and remote
 * models.
 */
public class RunnerBridgeProvider implements StreamingProvider {

    private final String runnerType;
    private final ModelRunnerFactory runnerFactory;
    private final CachedModelRepository modelRepository;

    public RunnerBridgeProvider(String runnerType, ModelRunnerFactory runnerFactory,
            CachedModelRepository modelRepository) {
        this.runnerType = runnerType;
        this.runnerFactory = runnerFactory;
        this.modelRepository = modelRepository;
    }

    @Override
    public String id() {
        return runnerType;
    }

    @Override
    public String name() {
        return "Local (" + runnerType + ")";
    }

    @Override
    public ProviderMetadata metadata() {
        return ProviderMetadata.builder()
                .providerId(id())
                .name(name())
                .version(version())
                .description("Local inference runner adapter for " + runnerType)
                .vendor("kayys-tech")
                .homepage("https://kayys.tech")
                .build();
    }

    @Override
    public ProviderCapabilities capabilities() {
        return ProviderCapabilities.builder()
                .streaming(true)
                .embeddings(true)
                .supportedFormats(java.util.Collections.emptySet())
                .maxContextTokens(32000)
                .build();
    }

    @Override
    public void initialize(ProviderConfig config) {
    }

    @Override
    public boolean supports(String modelId, ProviderRequest request) {
        String tenantId = resolveTenantId(request);
        try {
            ModelManifest manifest = modelRepository.findById(modelId, tenantId)
                    .await().atMost(Duration.ofSeconds(5));

            if (manifest == null) {
                return false;
            }

            String framework = extractFramework(runnerType);
            return manifest.artifacts().keySet().stream()
                    .anyMatch(format -> framework.equalsIgnoreCase(format.name()));

        } catch (Exception e) {
            return false;
        }
    }

    @Override
    public Uni<InferenceResponse> infer(ProviderRequest request) {
        String tenantId = resolveTenantId(request);
        // translate ProviderRequest to InferenceRequest (Engine's internal format)
        InferenceRequest inferenceRequest = InferenceRequest.builder()
                .requestId(request.getRequestId())
                .model(request.getModel())
                .messages(request.getMessages())
                .parameters(request.getParameters())
                .streaming(request.isStreaming())
                .timeout(request.getTimeout())
                .preferredProvider(runnerType)
                .build();

        return modelRepository
                .findById(request.getModel(), tenantId)
                .onItem().ifNull().failWith(() -> new ModelNotFoundException(request.getModel()))
                .chain(manifest -> {
                    ModelRunner runner = runnerFactory.getOrCreateRunner(runnerType, manifest);
                    try {
                        return Uni.createFrom().item(runner.infer(inferenceRequest));
                    } catch (tech.kayys.gollek.spi.exception.InferenceException e) {
                        return Uni.createFrom().failure(e);
                    }
                });
    }

    @Override
    public Multi<StreamChunk> inferStream(ProviderRequest request) {
        return Multi.createFrom().failure(new UnsupportedOperationException(
                "Streaming is not yet supported for local runners via RunnerBridgeProvider."));
    }

    @Override
    public Uni<ProviderHealth> health() {
        return Uni.createFrom().item(() -> ProviderHealth.healthy(runnerType));
    }

    @Override
    public void shutdown() {
    }

    private String extractFramework(String runnerName) {
        int dashIndex = runnerName.indexOf('-');
        if (dashIndex > 0) {
            return runnerName.substring(0, dashIndex);
        }
        return runnerName;
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
}