package tech.kayys.gollek.spi.auth;

/**
 * Shared API key constants for client/server authentication.
 */
public final class ApiKeyConstants {

    public static final String HEADER_API_KEY = "X-API-Key";
    public static final String HEADER_AUTHORIZATION = "Authorization";
    public static final String AUTHORIZATION_SCHEME = "ApiKey";
    public static final String COMMUNITY_API_KEY = "community";

    private ApiKeyConstants() {
    }

    public static String authorizationValue(String apiKey) {
        return AUTHORIZATION_SCHEME + " " + apiKey;
    }
}
