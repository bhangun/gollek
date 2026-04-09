package tech.kayys.gollek.sdk.core.tenant;

import java.util.Optional;

/**
 * Strategy for resolving tenant identification from the current runtime context.
 */
public interface TenantResolver {

    /**
     * Resolves the tenant ID for the current request.
     *
     * @return The resolved tenant ID, or empty if it cannot be determined
     */
    default Optional<String> resolveTenantId() {
        return Optional.ofNullable(resolveRequestId());
    }

    /**
     * Resolves the request ID for the current request context.
     * In most cases, this is synonymous with the organization or tenant ID.
     *
     * @return The resolved request ID
     */
    String resolveRequestId();

    /**
     * Resolves the API key to use for authenticated operations if no explicit key is provided.
     *
     * @return The default API key for the current context
     */
    String resolveApiKey();
}
