package tech.kayys.golek.sdk.factory;

import tech.kayys.golek.sdk.core.GolekSdk;
import tech.kayys.golek.sdk.local.LocalGolekSdk;

/**
 * Factory for creating Golek SDK instances.
 * Provides convenient methods to create local or remote SDK implementations.
 */
public class GolekSdkFactory {
    
    /**
     * Creates a local SDK instance that runs within the same JVM as the inference engine.
     * This is typically used when the SDK is embedded within the same application as the engine.
     * 
     * @return A local SDK instance
     */
    public static GolekSdk createLocalSdk() {
        // In a real implementation, this would properly initialize the CDI context
        // For now, we'll return a new instance directly
        return new LocalGolekSdk();
    }
    
    /**
     * Creates a remote SDK instance that communicates with the inference engine via HTTP API.
     * This is typically used when the SDK runs in a separate process or machine from the engine.
     * 
     * @param baseUrl The base URL of the inference engine API
     * @param apiKey The API key for authentication
     * @return A remote SDK instance
     */
    public static GolekSdk createRemoteSdk(String baseUrl, String apiKey) {
        return new tech.kayys.golek.client.GolekClient.Builder()
                .baseUrl(baseUrl)
                .apiKey(apiKey)
                .build();
    }
    
    /**
     * Creates a remote SDK instance with additional configuration options.
     * 
     * @param baseUrl The base URL of the inference engine API
     * @param apiKey The API key for authentication
     * @param defaultTenantId The default tenant ID to use
     * @return A remote SDK instance
     */
    public static GolekSdk createRemoteSdk(String baseUrl, String apiKey, String defaultTenantId) {
        return new tech.kayys.golek.client.GolekClient.Builder()
                .baseUrl(baseUrl)
                .apiKey(apiKey)
                .defaultTenantId(defaultTenantId)
                .build();
    }
}