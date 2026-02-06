package tech.kayys.golek.spi.provider;

import io.smallrye.mutiny.Multi;
import tech.kayys.golek.spi.stream.StreamChunk;
import tech.kayys.wayang.tenant.TenantContext;

/**
 * Extension for providers that support streaming output.
 */
public interface StreamingProvider extends LLMProvider {

    /**
     * Execute streaming inference request.
     * 
     * @param request Normalized inference request
     * @param context Tenant context
     * @return Multi with stream chunks
     */
    Multi<StreamChunk> inferStream(
            ProviderRequest request,
            TenantContext context);
}
