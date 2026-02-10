package tech.kayys.golek.sdk.factory;

import jakarta.enterprise.inject.se.SeContainer;
import jakarta.enterprise.inject.se.SeContainerInitializer;
import tech.kayys.golek.sdk.core.GolekSdk;
import tech.kayys.golek.sdk.core.config.SdkConfig;
import tech.kayys.golek.sdk.core.exception.SdkException;

import java.time.Duration;
import java.util.Objects;

/**
 * Factory for creating Golek SDK instances.
 * Provides convenient methods to create local or remote SDK implementations.
 *
 * <p>
 * Usage examples:
 * 
 * <pre>{@code
 * // Local SDK (embedded in same JVM as engine)
 * GolekSdk localSdk = GolekSdkFactory.createLocalSdk();
 *
 * // Local SDK with configuration
 * SdkConfig config = SdkConfig.builder()
 *         .apiKey("community")
 *         .preferredProvider("tech.kayys/ollama-provider")
 *         .build();
 * GolekSdk configuredSdk = GolekSdkFactory.createLocalSdk(config);
 *
 * // Remote SDK (communicates via HTTP)
 * GolekSdk remoteSdk = GolekSdkFactory.builder()
 *         .baseUrl("https://golek-spi.example.com")
 *         .apiKey("your-api-key")
 *         .buildRemote();
 * }</pre>
 */
public class GolekSdkFactory {

    private static volatile SeContainer cdiContainer;

    /**
     * Creates a local SDK instance that runs within the same JVM as the inference
     * engine.
     * Uses CDI to inject dependencies from the running Quarkus application.
     *
     * @return A local SDK instance
     * @throws SdkException if SDK creation fails
     */
    public static GolekSdk createLocalSdk() throws SdkException {
        // TODO: Implement proper CDI or service loading for LocalGolekSdk
        // For now, returning null to resolve compilation errors.
        // The correct approach would involve a service loader or CDI to find the
        // implementation
        return null;
    }

    /**
     * Creates a local SDK instance with custom configuration.
     *
     * @param config SDK configuration
     * @return A local SDK instance
     * @throws SdkException if SDK creation fails
     */
    public static GolekSdk createLocalSdk(SdkConfig config) throws SdkException {
        Objects.requireNonNull(config, "config cannot be null");

        GolekSdk sdk = createLocalSdk();
        if (sdk == null) {
            throw new SdkException("SDK_ERR_INIT", "Local SDK implementation not found or initialized.");
        }

        // Apply configuration
        config.getPreferredProvider().ifPresent(providerId -> {
            try {
                sdk.setPreferredProvider(providerId);
            } catch (Exception e) {
                throw new RuntimeException(new SdkException("SDK_ERR_CONFIG",
                        "Failed to set preferred provider: " + providerId, e));
            }
        });

        return sdk;
    }

    /**
     * Creates a remote SDK instance that communicates with the inference engine via
     * HTTP API.
     *
     * @param baseUrl The base URL of the inference engine API
     * @param apiKey  The API key for authentication
     * @return A remote SDK instance
     *         public static GolekSdk createRemoteSdk(String baseUrl, String apiKey)
     *         {
     *         return builder()
     *         .baseUrl(baseUrl)
     *         .apiKey(apiKey)
     *         .buildRemote();
     *         }
     * 
     *         /**
     *         Creates a remote SDK instance with additional configuration options.
     *
     * @param baseUrl  The base URL of the inference engine API
     * @param apiKey   The API key for authentication
     * @param apiKey The API key to use (defaults to community if not provided)
     * @return A remote SDK instance
     *         public static GolekSdk createRemoteSdk(String baseUrl, String apiKey,
     *         String apiKey) {
     *         return builder()
     *         .baseUrl(baseUrl)
     *         .apiKey(apiKey)
     *         .buildRemote();
     *         }
     * 
     *         /**
     *         Creates a remote SDK instance with custom configuration.
     *
     * @param config SDK configuration
     * @return A remote SDK instance
     *         public static GolekSdk createRemoteSdk(SdkConfig config) {
     *         Objects.requireNonNull(config, "config cannot be null");
     * 
     *         Builder builder = builder()
     *         .apiKey(config.getApiKey())
     *         .requestTimeout(config.getRequestTimeout())
     *         .requestTimeout(config.getConnectTimeout())
     *         .maxRetries(config.getRetryConfig().getMaxAttempts());
     * 
     *         config.getPreferredProvider().ifPresent(builder::preferredProvider);
     * 
     *         // Note: baseUrl and apiKey should be in provider configs for remote
     *         // This is a simplified implementation
     *         return builder.buildRemote();
     *         }
     * 
     *         /**
     *         Creates a new builder for configuring SDK instances.
     *
     * @return A new builder instance
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Gets or initializes the CDI container for local SDK instances.
     */
    private static SeContainer getCdiContainer() {
        if (cdiContainer == null) {
            synchronized (GolekSdkFactory.class) {
                if (cdiContainer == null) {
                    try {
                        // Try to initialize CDI container
                        cdiContainer = SeContainerInitializer.newInstance().initialize();
                    } catch (Exception e) {
                        // CDI not available, will use direct instantiation
                        return null;
                    }
                }
            }
        }
        return cdiContainer;
    }

    /**
     * Shuts down the CDI container if it was initialized.
     * Should be called when the application is shutting down.
     */
    public static void shutdown() {
        if (cdiContainer != null) {
            synchronized (GolekSdkFactory.class) {
                if (cdiContainer != null) {
                    cdiContainer.close();
                    cdiContainer = null;
                }
            }
        }
    }

    /**
     * Builder for creating configured SDK instances.
     */
    public static class Builder {
        private String baseUrl;
        private String apiKey = "community";
        private String preferredProvider;
        private Duration requestTimeout = Duration.ofSeconds(60);
        private Duration connectTimeout = Duration.ofSeconds(30);
        private int maxRetries = 3;
        private boolean enableMetrics = false;

        private Builder() {
        }

        public Builder baseUrl(String baseUrl) {
            this.baseUrl = Objects.requireNonNull(baseUrl, "baseUrl cannot be null");
            return this;
        }

        public Builder apiKey(String apiKey) {
            this.apiKey = Objects.requireNonNull(apiKey, "apiKey cannot be null");
            return this;
        }

        public Builder preferredProvider(String preferredProvider) {
            this.preferredProvider = preferredProvider;
            return this;
        }

        public Builder requestTimeout(Duration requestTimeout) {
            this.requestTimeout = Objects.requireNonNull(requestTimeout, "requestTimeout cannot be null");
            return this;
        }

        public Builder connectTimeout(Duration connectTimeout) {
            this.connectTimeout = Objects.requireNonNull(connectTimeout, "connectTimeout cannot be null");
            return this;
        }

        public Builder maxRetries(int maxRetries) {
            if (maxRetries < 0) {
                throw new IllegalArgumentException("maxRetries must be >= 0");
            }
            this.maxRetries = maxRetries;
            return this;
        }

        public Builder enableMetrics(boolean enableMetrics) {
            this.enableMetrics = enableMetrics;
            return this;
        }

        /**
         * Builds a local SDK instance with the configured settings.
         *
         * @return A configured local SDK instance
         */
        public GolekSdk buildLocal() {
            // TODO: Implement proper CDI or service loading for LocalGolekSdk
            // For now, returning null to resolve compilation errors.
            // The correct approach would involve a service loader or CDI to find the
            // implementation
            return null;
        }

        /**
         * Builds a remote SDK instance with the configured settings.
         *
         * @return A configured remote SDK instance
         * @throws SdkException if configuration is invalid
         */
        public GolekSdk buildRemote() throws SdkException {
            if (baseUrl == null || baseUrl.isBlank()) {
                throw new SdkException("SDK_ERR_CONFIG", "baseUrl is required for remote SDK");
            }
            if (apiKey == null || apiKey.isBlank()) {
                apiKey = "community";
            }
            // TODO: Implement proper remote SDK instantiation when golek-sdk-java-remote is
            // available
            throw new UnsupportedOperationException("Remote SDK not yet implemented.");
        }
    }
}
