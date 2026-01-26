package tech.kayys.golek.provider.core.spi;

import io.smallrye.mutiny.Multi;
import tech.kayys.golek.api.provider.ProviderRequest;
import tech.kayys.golek.api.stream.StreamChunk;

/**
 * Extension interface for providers that support streaming
 */
public interface StreamingProvider extends LLMProvider {

    /**
     * Execute streaming inference
     * Returns a reactive stream of chunks
     */
    Multi<StreamChunk> stream(ProviderRequest request);

}