package tech.kayys.gollek.sdk.factory;

import jakarta.enterprise.inject.se.SeContainer;
import jakarta.enterprise.inject.se.SeContainerInitializer;
import jakarta.enterprise.inject.spi.CDI;
import tech.kayys.gollek.sdk.core.GollekSdk;
import tech.kayys.gollek.sdk.core.config.SdkConfig;
import tech.kayys.gollek.sdk.core.exception.SdkException;

import java.time.Duration;
import java.util.Objects;

/**
 * Factory for creating Gollek SDK instances.
 * Provides convenient methods to create local or remote SDK implementations.
 *
 * <p>
 * Usage examples:
 * 
 * <pre>{@code
 * // Local SDK (embedded in same JVM as engine)
 * GollekSdk localSdk = GollekSdkFactory.createLocalSdk();
 *
 * // Local SDK with configuration
 * SdkConfig config = SdkConfig.builder()
 *         .apiKey("community")
 *         .preferredProvider("tech.kayys/ollama-provider")
 *         .build();
 * GollekSdk configuredSdk = GollekSdkFactory.createLocalSdk(config);
 *
 * // Remote SDK (communicates via HTTP)
 * GollekSdk remoteSdk = GollekSdkFactory.builder()
 *         .baseUrl("https://gollek-spi.example.com")
 *         .apiKey("your-api-key")
 *         .buildRemote();
 * }</pre>
 */
public class GollekSdkFactory {

    private static volatile SeContainer cdiContainer;

    /**
     * Creates a local SDK instance that runs within the same JVM as the inference
     * engine.
     * Uses CDI to inject dependencies from the running Quarkus application.
     *
     * @return A local SDK instance
     * @throws SdkException if SDK creation fails
     */
    public static GollekSdk createLocalSdk() throws SdkException {
        GollekSdk sdk = resolveFromCurrentCdi();
        if (sdk != null) {
            return sdk;
        }

        SeContainer container = getCdiContainer();
        if (container != null) {
            var selection = container.select(GollekSdk.class);
            if (selection.isResolvable()) {
                return selection.get();
            }
        }

        throw new SdkException("SDK_ERR_INIT",
                "Local SDK implementation not found. Ensure gollek-sdk-java-local is on the classpath and CDI is active.");
    }

    /**
     * Creates a local SDK instance with custom configuration.
     *
     * @param config SDK configuration
     * @return A local SDK instance
     * @throws SdkException if SDK creation fails
     */
    public static GollekSdk createLocalSdk(SdkConfig config) throws SdkException {
        Objects.requireNonNull(config, "config cannot be null");

        GollekSdk sdk = createLocalSdk();
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
     */
    public static GollekSdk createRemoteSdk(String baseUrl, String apiKey) throws SdkException {
         return builder()
         .baseUrl(baseUrl)
         .apiKey(apiKey)
         .buildRemote();
    }

    /**
     * Creates a remote SDK instance with custom configuration.
     *
     * @param config SDK configuration
     * @return A remote SDK instance
     */
    public static GollekSdk createRemoteSdk(SdkConfig config) throws SdkException {
        Objects.requireNonNull(config, "config cannot be null");

        Builder builder = builder()
        .apiKey(config.getApiKey())
        .requestTimeout(config.getRequestTimeout())
        .connectTimeout(config.getConnectTimeout())
        .maxRetries(config.getRetryConfig().getMaxAttempts());

        config.getPreferredProvider().ifPresent(builder::preferredProvider);

        return builder.buildRemote();
    }

    /**
     * Creates a new builder for configuring SDK instances.
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
            synchronized (GollekSdkFactory.class) {
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

    private static GollekSdk resolveFromCurrentCdi() {
        try {
            var cdi = CDI.current();
            var selection = cdi.select(GollekSdk.class);
            if (selection.isResolvable()) {
                return selection.get();
            }
        } catch (IllegalStateException ignored) {
            // No active CDI context in this runtime; fallback to SeContainer initialization.
        } catch (Exception ignored) {
            // CDI API unavailable or bean resolution failed; fallback path below.
        }
        return null;
    }

    /**
     * Shuts down the CDI container if it was initialized.
     * Should be called when the application is shutting down.
     */
    public static void shutdown() {
        if (cdiContainer != null) {
            synchronized (GollekSdkFactory.class) {
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
        public GollekSdk buildLocal() {
            try {
                SdkConfig config = SdkConfig.builder()
                        .apiKey(apiKey)
                        .requestTimeout(requestTimeout)
                        .connectTimeout(connectTimeout)
                        .preferredProvider(preferredProvider)
                        .enableMetrics(enableMetrics)
                        .build();
                return createLocalSdk(config);
            } catch (SdkException e) {
                throw new IllegalStateException("Failed to create local SDK", e);
            }
        }

        /**
         * Builds a remote SDK instance with the configured settings.
         *
         * @return A configured remote SDK instance
         * @throws SdkException if configuration is invalid
         */
        public GollekSdk buildRemote() throws SdkException {
            if (baseUrl == null || baseUrl.isBlank()) {
                throw new SdkException("SDK_ERR_CONFIG", "baseUrl is required for remote SDK");
            }
            if (apiKey == null || apiKey.isBlank()) {
                apiKey = "community";
            }
            // TODO: Implement proper remote SDK instantiation when gollek-sdk-java-remote is
            // available
            throw new UnsupportedOperationException("Remote SDK not yet implemented.");
        }
    }
}
