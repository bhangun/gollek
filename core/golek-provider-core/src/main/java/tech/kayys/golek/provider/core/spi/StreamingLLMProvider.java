package tech.kayys.golek.provider.core.spi;

import io.smallrye.mutiny.Multi;
import tech.kayys.golek.api.provider.ProviderRequest;
import tech.kayys.golek.api.stream.StreamChunk;

/**
 * Provider with streaming support (SSE/WebSocket).
 */
public interface StreamingLLMProvider extends LLMProvider {

    /**
     * Execute streaming inference
     */
    Multi<StreamChunk> stream(ProviderRequest request);
}