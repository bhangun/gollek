package tech.kayys.gollek.runtime.standalone;

import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.provider.LLMProvider;
import tech.kayys.gollek.spi.provider.ProviderConfig;
import tech.kayys.gollek.spi.provider.ProviderRequest;
import tech.kayys.gollek.spi.provider.ProviderHealth;
import tech.kayys.gollek.spi.provider.ProviderMetadata;
import tech.kayys.gollek.spi.provider.ProviderCapabilities;
import tech.kayys.gollek.spi.exception.ProviderException;

import io.smallrye.mutiny.Multi;
import tech.kayys.gollek.spi.provider.StreamingProvider;
import tech.kayys.gollek.spi.stream.StreamChunk;

@ApplicationScoped
public class MockProvider implements StreamingProvider {

    @Override
    public String id() {
        return "mock";
    }

    @Override
    public String name() {
        return "Mock Provider";
    }

    @Override
    public ProviderMetadata metadata() {
        return ProviderMetadata.builder()
                .providerId("mock")
                .name("mock")
                .version("1.0")
                .build();
    }

    @Override
    public ProviderCapabilities capabilities() {
        return ProviderCapabilities.builder().build();
    }

    @Override
    public void initialize(ProviderConfig config) throws ProviderException.ProviderInitializationException {
    }

    @Override
    public boolean supports(String modelId, ProviderRequest request) {
        return "test-model".equals(modelId);
    }

    @Override
    public Uni<InferenceResponse> infer(ProviderRequest request) {
        return Uni.createFrom().item(InferenceResponse.builder()
                .requestId(request.getRequestId() != null ? request.getRequestId() : "test-req")
                .content("Hello from mock!")
                .model("test-model")
                .tokensUsed(10)
                .build());
    }

    @Override
    public Uni<ProviderHealth> health() {
        return Uni.createFrom().item(ProviderHealth.healthy("mock"));
    }

    @Override
    public Multi<StreamChunk> inferStream(ProviderRequest request) {
        return Multi.createFrom().items(
                StreamChunk.of(request.getRequestId(), 0, "Hello "),
                StreamChunk.of(request.getRequestId(), 1, "from "),
                StreamChunk.of(request.getRequestId(), 2, "mock "),
                StreamChunk.of(request.getRequestId(), 3, "stream!"),
                StreamChunk.finalChunk(request.getRequestId(), 4, ""));
    }

    @Override
    public void shutdown() {
    }
}
