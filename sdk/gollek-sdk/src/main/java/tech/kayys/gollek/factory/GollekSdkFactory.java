package tech.kayys.gollek.factory;

import tech.kayys.gollek.sdk.config.SdkConfig;
import tech.kayys.gollek.sdk.core.GollekSdk;
import tech.kayys.gollek.sdk.core.GollekSdkProvider;
import tech.kayys.gollek.sdk.exception.SdkException;

import java.time.Duration;
import java.util.Comparator;
import java.util.Objects;
import java.util.Optional;
import java.util.ServiceLoader;

/**
 * Factory for creating Gollek SDK instances.
 * Discovers implementations via {@link ServiceLoader} — no hard dependency
 * on local or remote modules.
 *
 * <p>
 * Usage examples:
 *
 * <pre>{@code
 * // Local SDK (embedded in same JVM as engine)
 * // Requires gollek-sdk-java-local on the classpath
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
 * // Requires gollek-sdk-java-remote on the classpath
 * GollekSdk remoteSdk = GollekSdkFactory.builder()
 *         .baseUrl("https://gollek-spi.example.com")
 *         .apiKey("your-api-key")
 *         .buildRemote();
 *
 * // Or use the implementation modules directly without this factory:
 * // - LocalGollekSdk (via CDI injection)
 * // - GollekClient.builder().baseUrl("...").build()
 * }</pre>
 */
public class GollekSdkFactory {

    private GollekSdkFactory() {
    }

    /**
     * Creates a local SDK instance that runs within the same JVM as the inference
     * engine.
     * Discovers the local provider via ServiceLoader.
     *
     * @return A local SDK instance
     * @throws SdkException if SDK creation fails or no local provider is on the
     *                      classpath
     */
    public static GollekSdk createLocalSdk() throws SdkException {
        return createLocalSdk(null);
    }

    /**
     * Creates a local SDK instance with custom configuration.
     *
     * @param config SDK configuration (may be null for defaults)
     * @return A local SDK instance
     * @throws SdkException if SDK creation fails
     */
    public static GollekSdk createLocalSdk(SdkConfig config) throws SdkException {
        GollekSdkProvider provider = findProvider(GollekSdkProvider.Mode.LOCAL)
                .orElseThrow(() -> new SdkException("SDK_ERR_INIT",
                        "No local SDK provider found. Add gollek-sdk-java-local to your classpath."));
        return provider.create(config);
    }

    /**
     * Creates a remote SDK instance that communicates with the inference engine via
     * HTTP API.
     *
     * @param baseUrl The base URL of the inference engine API
     * @param apiKey  The API key for authentication
     * @return A remote SDK instance
     * @throws SdkException if SDK creation fails
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
     * @throws SdkException if SDK creation fails
     */
    public static GollekSdk createRemoteSdk(SdkConfig config) throws SdkException {
        Objects.requireNonNull(config, "config cannot be null");
        GollekSdkProvider provider = findProvider(GollekSdkProvider.Mode.REMOTE)
                .orElseThrow(() -> new SdkException("SDK_ERR_INIT",
                        "No remote SDK provider found. Add gollek-sdk-java-remote to your classpath."));
        return provider.create(config);
    }

    /**
     * Auto-detect the best available SDK from the classpath.
     * Prefers LOCAL over REMOTE when both are available.
     *
     * @return An SDK instance
     * @throws SdkException if no provider is found
     */
    public static GollekSdk create() throws SdkException {
        return create(null);
    }

    /**
     * Auto-detect the best available SDK from the classpath with configuration.
     *
     * @param config SDK configuration (may be null for defaults)
     * @return An SDK instance
     * @throws SdkException if no provider is found
     */
    public static GollekSdk create(SdkConfig config) throws SdkException {
        GollekSdkProvider provider = ServiceLoader.load(GollekSdkProvider.class)
                .stream()
                .map(ServiceLoader.Provider::get)
                .min(Comparator.comparingInt(GollekSdkProvider::priority))
                .orElseThrow(() -> new SdkException("SDK_ERR_INIT",
                        "No SDK provider found. Add gollek-sdk-java-local or gollek-sdk-java-remote to your classpath."));
        return provider.create(config);
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
     * Finds a provider for the given mode via ServiceLoader.
     */
    private static Optional<GollekSdkProvider> findProvider(GollekSdkProvider.Mode mode) {
        return ServiceLoader.load(GollekSdkProvider.class)
                .stream()
                .map(ServiceLoader.Provider::get)
                .filter(p -> p.mode() == mode)
                .min(Comparator.comparingInt(GollekSdkProvider::priority));
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

        private SdkConfig toConfig() {
            return SdkConfig.builder()
                    .apiKey(apiKey)
                    .requestTimeout(requestTimeout)
                    .connectTimeout(connectTimeout)
                    .preferredProvider(preferredProvider)
                    .enableMetrics(enableMetrics)
                    .maxRetries(maxRetries)
                    .build();
        }

        /**
         * Builds a local SDK instance with the configured settings.
         *
         * @return A configured local SDK instance
         * @throws SdkException if the local provider is not available
         */
        public GollekSdk buildLocal() throws SdkException {
            return createLocalSdk(toConfig());
        }

        /**
         * Builds a remote SDK instance with the configured settings.
         *
         * @return A configured remote SDK instance
         * @throws SdkException if configuration is invalid or remote provider not
         *                      available
         */
        public GollekSdk buildRemote() throws SdkException {
            if (baseUrl == null || baseUrl.isBlank()) {
                throw new SdkException("SDK_ERR_CONFIG", "baseUrl is required for remote SDK");
            }
            return createRemoteSdk(toConfig());
        }

        /**
         * Auto-detect the best available SDK from the classpath.
         *
         * @return A configured SDK instance
         * @throws SdkException if no provider is found
         */
        public GollekSdk build() throws SdkException {
            return create(toConfig());
        }
    }
}
