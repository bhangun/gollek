package tech.kayys.golek.provider.core.streaming;

import io.smallrye.mutiny.Multi;
import tech.kayys.golek.api.provider.ProviderRequest;
import tech.kayys.golek.api.tenant.TenantContext; // Added
import tech.kayys.golek.api.stream.StreamChunk;
import tech.kayys.golek.provider.core.spi.LLMProvider;

/**
 * Provider that supports streaming responses.
 */
public interface StreamingLLMProvider extends LLMProvider {

    /**
     * Execute streaming inference.
     * 
     * @param request ProviderRequest
     * @param context TenantContext
     * @return Multi<StreamChunk>
     */
    Multi<StreamChunk> stream(
            ProviderRequest request,
            TenantContext context); // Add context parameter to match usage

    /**
     * Check if streaming is enabled for this provider instance
     */
    default boolean isStreamingEnabled() {
        return capabilities().isStreaming();
    }
}
