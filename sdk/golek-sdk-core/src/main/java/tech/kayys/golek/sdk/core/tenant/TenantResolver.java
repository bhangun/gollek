package tech.kayys.golek.sdk.core.tenant;

/**
 * Interface for resolving the current API key.
 * Implementations can extract API key from security context, headers, or other sources.
 */
public interface TenantResolver {

    /**
     * Resolves the current API key.
     *
     * @return The API key, never null
     * @throws IllegalStateException if API key cannot be resolved
     */
    default String resolveApiKey() {
        return resolveTenantId();
    }

    /**
     * @deprecated Use {@link #resolveApiKey()}.
     */
    @Deprecated
    String resolveTenantId();

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
            public String resolveTenantId() {
                return apiKey;
            }
        };
    }

    /**
     * @deprecated Use {@link #fixedApiKey(String)}.
     */
    @Deprecated
    static TenantResolver fixed(String tenantId) {
        return fixedApiKey(tenantId);
    }
}
