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

@ApplicationScoped
public class MockProvider implements LLMProvider {

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
        return "tinyllama".equals(modelId);
    }

    @Override
    public Uni<InferenceResponse> infer(ProviderRequest request) {
        return Uni.createFrom().item(InferenceResponse.builder()
                .requestId(request.getRequestId() != null ? request.getRequestId() : "test-req")
                .content("Hello from mock!")
                .model("tinyllama")
                .tokensUsed(10)
                .build());
    }

    @Override
    public Uni<ProviderHealth> health() {
        return Uni.createFrom().item(ProviderHealth.healthy("mock"));
    }

    @Override
    public void shutdown() {
    }
}
