package tech.kayys.gollek.engine.adapter;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;

import tech.kayys.gollek.spi.provider.ProviderCapabilities;
import tech.kayys.gollek.spi.provider.ProviderConfig;
import tech.kayys.gollek.spi.provider.ProviderHealth;
import tech.kayys.gollek.spi.provider.ProviderMetadata;
import tech.kayys.gollek.spi.provider.ProviderRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;
import tech.kayys.gollek.spi.provider.StreamingProvider;

/**
 * Stub provider adapter - to be properly implemented.
 * Bridges local ModelRunner to Provider interface.
 */
public class RunnerBridgeProvider implements StreamingProvider {

    private final String runnerType;

    public RunnerBridgeProvider(String runnerType) {
        this.runnerType = runnerType;
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
                .version("1.0.0")
                .description("Local inference runner adapter for " + runnerType)
                .build();
    }

    @Override
    public ProviderCapabilities capabilities() {
        return ProviderCapabilities.builder()
                .streaming(false)
                .build();
    }

    @Override
    public void initialize(ProviderConfig config) {
        // No-op
    }

    @Override
    public boolean supports(String modelId, ProviderRequest request) {
        return true; // Default implementation
    }

    @Override
    public Uni<InferenceResponse> infer(ProviderRequest request) {
        return Uni.createFrom().failure(
                new UnsupportedOperationException("Not yet implemented"));
    }

    @Override
    public Multi<StreamingInferenceChunk> inferStream(ProviderRequest request) {
        return Multi.createFrom().failure(
                new UnsupportedOperationException("Streaming not supported"));
    }

    @Override
    public Uni<ProviderHealth> health() {
        return Uni.createFrom().item(ProviderHealth.healthy());
    }

    @Override
    public void shutdown() {
        // No-op
    }
}
