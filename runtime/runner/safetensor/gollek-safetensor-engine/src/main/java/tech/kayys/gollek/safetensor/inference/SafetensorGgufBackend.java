package tech.kayys.gollek.inference.safetensor;

import io.smallrye.mutiny.Multi;
import io.smallrye.mutiny.Uni;
import tech.kayys.gollek.spi.exception.ProviderException;
import tech.kayys.gollek.spi.provider.ProviderCapabilities;
import tech.kayys.gollek.spi.provider.ProviderConfig;
import tech.kayys.gollek.spi.provider.ProviderHealth;
import tech.kayys.gollek.spi.provider.ProviderRequest;
import tech.kayys.gollek.spi.inference.InferenceResponse;
import tech.kayys.gollek.spi.inference.StreamingInferenceChunk;

interface SafetensorGgufBackend {
    void initialize(ProviderConfig config) throws ProviderException.ProviderInitializationException;

    Uni<InferenceResponse> infer(ProviderRequest request);

    Multi<StreamingInferenceChunk> inferStream(ProviderRequest request);

    Uni<ProviderHealth> health();

    ProviderCapabilities capabilities();
}
