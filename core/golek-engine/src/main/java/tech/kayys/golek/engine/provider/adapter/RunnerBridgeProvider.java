package tech.kayys.golek.engine.provider.adapter;

import io.smallrye.mutiny.Uni;
import tech.kayys.golek.api.inference.InferenceRequest;
import tech.kayys.golek.api.inference.InferenceResponse;
import tech.kayys.golek.api.model.ModelManifest;
import tech.kayys.golek.api.provider.ProviderCapabilities;
import tech.kayys.golek.api.provider.ProviderConfig;
import tech.kayys.golek.api.provider.ProviderHealth;
import tech.kayys.golek.api.provider.ProviderMetadata;
import tech.kayys.golek.api.provider.ProviderRequest;
import tech.kayys.golek.engine.execution.ModelRunnerFactory;
import tech.kayys.golek.engine.model.ModelRunner;
import tech.kayys.golek.model.exception.ModelNotFoundException;
import tech.kayys.golek.engine.model.ModelRepository;
import tech.kayys.golek.provider.core.spi.LLMProvider;
import tech.kayys.golek.provider.core.spi.StreamingProvider;
import tech.kayys.wayang.tenant.TenantContext;
import tech.kayys.golek.api.stream.StreamChunk;
import io.smallrye.mutiny.Multi;

/**
 * Adapter that exposes a local ModelRunner as an LLMProvider.
 * This enables the unified routing and execution of both local and remote
 * models.
 */
public class RunnerBridgeProvider implements LLMProvider, StreamingProvider {

    private final String runnerType;
    private final ModelRunnerFactory runnerFactory;
    private final ModelRepository modelRepository;

    public RunnerBridgeProvider(String runnerType, ModelRunnerFactory runnerFactory, ModelRepository modelRepository) {
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
        // e.g. "litert-cpu" -> "Local (LiteRT CPU)"
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
        // Capabilities depend on the underlying runner, which we can't fully know
        // without instantiating it.
        // For now, we return a broad set of capabilities and rely on supports() and
        // runtime checks.
        // Ideally, ModelRunnerFactory would expose capabilities per runner type.
        return ProviderCapabilities.builder()
                .streaming(true) // Most local runners support streaming
                .embeddings(true)
                .supportedFormats(java.util.Collections.emptySet()) // Dynamic
                .maxContextTokens(32000) // Placeholder
                .build();
    }

    @Override
    public void initialize(ProviderConfig config) {
        // No heavy initialization here, as runners are lazy-loaded by the factory
    }

    @Override
    public boolean supports(String modelId, TenantContext tenantContext) {
        // We need to check if the underlying runner supports the model's format.
        // This requires looking up the model manifest.
        try {
            ModelManifest manifest = modelRepository.findById(modelId, tenantContext.getTenantId())
                    .orElse(null);

            if (manifest == null) {
                return false;
            }

            // Ideally ModelRunnerFactory or ModelRunner should have a static/lightweight
            // supports() check.
            // For now, we rely on the logic that ModelRouter used: check format
            // compatibility.
            // But checking that here requires duplicating logic or exposing it from
            // Factory.
            // Let's assume if the factory has this runner, it *might* support it.
            // A more robust check would involve checking manifest formats against runner's
            // supported formats.

            // Re-using logic similar to ModelRouter's extractFramework approach:
            String framework = extractFramework(runnerType);
            // Check if any of the artifacts match the framework
            // Assuming framework string maps to ModelFormat logic.
            // A more precise check would convert string framework to ModelFormat if
            // possible.
            // For now, we check if any artifact's format name matches the framework string.
            return manifest.artifacts().keySet().stream()
                    .anyMatch(format -> framework.equalsIgnoreCase(format.name()));

        } catch (Exception e) {
            return false;
        }
    }

    @Override
    public Uni<InferenceResponse> infer(ProviderRequest request, TenantContext context) {
        return Uni.createFrom().item(() -> {
            try {
                // translate ProviderRequest to InferenceRequest (Engine's internal format)
                // Note: ProviderRequest and InferenceRequest are very similar.
                // We might need a converter. For now, assuming manual mapping.

                InferenceRequest inferenceRequest = InferenceRequest.builder()
                        .requestId(request.getRequestId())
                        .model(request.getModel())
                        .messages(request.getMessages())
                        .tools(request.getTools())
                        .toolChoice(request.getToolChoice())
                        .parameters(request.getParameters())
                        .streaming(request.isStreaming())
                        .timeout(request.getTimeout())
                        .priority(5) // Default priority
                        .preferredProvider(runnerType) // Lock to this runner
                        .build();

                // Get Manifest
                ModelManifest manifest = modelRepository.findById(request.getModel(), context.getTenantId())
                        .orElseThrow(() -> new ModelNotFoundException(request.getModel()));

                // Get Runner
                ModelRunner runner = runnerFactory.getOrCreateRunner(runnerType, manifest);

                // Execute logic matching InferenceOrchestrator's executeWithRunner
                // Note: ModelRunner.infer() is synchronous or async.

                return runner.infer(inferenceRequest);

            } catch (Exception e) {
                // Wrap in RuntimeException for Uni
                throw new RuntimeException(e);
            }
        });
    }

    @Override
    public Multi<StreamChunk> stream(ProviderRequest request, TenantContext context) {
        // Currently ModelRunner does not have a native stream() method.
        // We throw an exception until ModelRunner SPI is updated to support streaming.
        return Multi.createFrom().failure(new UnsupportedOperationException(
                "Streaming is not yet supported for local runners via RunnerBridgeProvider. " +
                        "Underlying ModelRunner SPI needs to be updated."));
    }

    @Override
    public Uni<ProviderHealth> health() {
        return Uni.createFrom().item(() -> {
            // Check factory pool status or just report healthy if factory is up
            return ProviderHealth.healthy(runnerType);
        });
    }

    @Override
    public void shutdown() {
        // Nothing to do, factory manages lifecycle
    }

    private String extractFramework(String runnerName) {
        int dashIndex = runnerName.indexOf('-');
        if (dashIndex > 0) {
            return runnerName.substring(0, dashIndex);
        }
        return runnerName;
    }
}
