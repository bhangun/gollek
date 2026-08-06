package tech.kayys.gollek.inference.safetensor;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import tech.kayys.gollek.spi.exception.ProviderException;
import tech.kayys.gollek.inference.gguf.GGUFProvider;
import tech.kayys.gollek.spi.provider.ProviderCapabilities;
import tech.kayys.gollek.spi.provider.ProviderConfig;
import tech.kayys.gollek.spi.provider.ProviderHealth;
import tech.kayys.gollek.spi.provider.ProviderRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;

@ApplicationScoped
public class GgufSafetensorBackend implements SafetensorGgufBackend {

    @Inject
    GGUFProvider ggufProvider;

    @Override
    public void initialize(ProviderConfig config) throws ProviderException.ProviderInitializationException {
        ggufProvider.initialize(config);
    }

    @Override
    public Uni<InferenceResponse> infer(ProviderRequest request) {
        return ggufProvider.infer(request);
    }

    @Override
    public Multi<StreamingInferenceChunk> inferStream(ProviderRequest request) {
        return ggufProvider.inferStream(request);
    }

    @Override
    public Uni<ProviderHealth> health() {
        return ggufProvider.health();
    }

    @Override
    public ProviderCapabilities capabilities() {
        return ggufProvider.capabilities();
    }
}
