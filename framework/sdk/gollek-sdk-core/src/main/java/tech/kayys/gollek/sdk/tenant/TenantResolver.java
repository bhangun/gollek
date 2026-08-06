package tech.kayys.gollek.sdk.tenant;

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
    String resolveApiKey();

    /**
     * Default implementation that returns a fixed API key.
     */
    static TenantResolver fixedApiKey(String apiKey) {
        return new TenantResolver() {
            @Override
            public String resolveApiKey() {
                return apiKey;
            }
        };
    }
}
