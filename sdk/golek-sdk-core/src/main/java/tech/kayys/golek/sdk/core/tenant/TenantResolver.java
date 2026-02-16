package tech.kayys.golek.sdk.core.tenant;

/**
 * Interface for resolving the current API key.
 * Implementations can extract API key from security context, headers, or other
 * sources.
 */
public interface TenantResolver {

    /**
     * Resolves the current API key.
     *
     * @return The API key, never null
     * @throws IllegalStateException if API key cannot be resolved
     */
    default String resolveApiKey() {
        return resolveRequestId();
    }

    /**
     * @deprecated Use {@link #resolveApiKey()}.
     */
    @Deprecated
    String resolveRequestId();

    /**
     * Default implementation that returns a fixed API key.
     */
    static TenantResolver fixedApiKey(String apiKey) {
        return new TenantResolver() {
            @Override
            public String resolveApiKey() {
                return apiKey;
            }

            @Override
            public String resolveRequestId() {
                return apiKey;
            }
        };
    }

    /**
     * @deprecated Use {@link #fixedApiKey(String)}.
     */
    @Deprecated
    static TenantResolver fixed(String requestId) {
        return fixedApiKey(requestId);
    }
}
