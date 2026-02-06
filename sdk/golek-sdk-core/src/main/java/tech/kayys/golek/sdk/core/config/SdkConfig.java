package tech.kayys.golek.sdk.core.config;

import java.time.Duration;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

/**
 * Global configuration for the Golek SDK.
 */
public class SdkConfig {
    private final String tenantId;
    private final String preferredProvider;
    private final Map<String, ProviderConfig> providerConfigs;
    private final Duration requestTimeout;
    private final Duration connectTimeout;
    private final RetryConfig retryConfig;
    private final boolean enableMetrics;
    private final boolean enableCircuitBreaker;

    private SdkConfig(Builder builder) {
        this.tenantId = builder.tenantId;
        this.preferredProvider = builder.preferredProvider;
        this.providerConfigs = new HashMap<>(builder.providerConfigs);
        this.requestTimeout = builder.requestTimeout;
        this.connectTimeout = builder.connectTimeout;
        this.retryConfig = builder.retryConfig;
        this.enableMetrics = builder.enableMetrics;
        this.enableCircuitBreaker = builder.enableCircuitBreaker;
    }

    public String getTenantId() {
        return tenantId;
    }

    public Optional<String> getPreferredProvider() {
        return Optional.ofNullable(preferredProvider);
    }

    public Optional<ProviderConfig> getProviderConfig(String providerId) {
        return Optional.ofNullable(providerConfigs.get(providerId));
    }

    public Map<String, ProviderConfig> getAllProviderConfigs() {
        return new HashMap<>(providerConfigs);
    }

    public Duration getRequestTimeout() {
        return requestTimeout;
    }

    public Duration getConnectTimeout() {
        return connectTimeout;
    }

    public RetryConfig getRetryConfig() {
        return retryConfig;
    }

    public boolean isEnableMetrics() {
        return enableMetrics;
    }

    public boolean isEnableCircuitBreaker() {
        return enableCircuitBreaker;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private String tenantId = "default";
        private String preferredProvider;
        private Map<String, ProviderConfig> providerConfigs = new HashMap<>();
        private Duration requestTimeout = Duration.ofSeconds(60);
        private Duration connectTimeout = Duration.ofSeconds(30);
        private RetryConfig retryConfig = RetryConfig.builder().build();
        private boolean enableMetrics = false;
        private boolean enableCircuitBreaker = true;

        public Builder tenantId(String tenantId) {
            this.tenantId = tenantId;
            return this;
        }

        public Builder preferredProvider(String preferredProvider) {
            this.preferredProvider = preferredProvider;
            return this;
        }

        public Builder providerConfig(String providerId, ProviderConfig config) {
            this.providerConfigs.put(providerId, config);
            return this;
        }

        public Builder requestTimeout(Duration requestTimeout) {
            this.requestTimeout = requestTimeout;
            return this;
        }

        public Builder connectTimeout(Duration connectTimeout) {
            this.connectTimeout = connectTimeout;
            return this;
        }

        public Builder retryConfig(RetryConfig retryConfig) {
            this.retryConfig = retryConfig;
            return this;
        }

        public Builder enableMetrics(boolean enableMetrics) {
            this.enableMetrics = enableMetrics;
            return this;
        }

        public Builder enableCircuitBreaker(boolean enableCircuitBreaker) {
            this.enableCircuitBreaker = enableCircuitBreaker;
            return this;
        }

        public SdkConfig build() {
            return new SdkConfig(this);
        }
    }

    /**
     * Configuration for a specific provider
     */
    public static class ProviderConfig {
        private final String apiKey;
        private final String endpoint;
        private final Map<String, String> additionalParams;

        private ProviderConfig(ProviderConfigBuilder builder) {
            this.apiKey = builder.apiKey;
            this.endpoint = builder.endpoint;
            this.additionalParams = new HashMap<>(builder.additionalParams);
        }

        public Optional<String> getApiKey() {
            return Optional.ofNullable(apiKey);
        }

        public Optional<String> getEndpoint() {
            return Optional.ofNullable(endpoint);
        }

        public Map<String, String> getAdditionalParams() {
            return new HashMap<>(additionalParams);
        }

        public static ProviderConfigBuilder builder() {
            return new ProviderConfigBuilder();
        }

        public static class ProviderConfigBuilder {
            private String apiKey;
            private String endpoint;
            private Map<String, String> additionalParams = new HashMap<>();

            public ProviderConfigBuilder apiKey(String apiKey) {
                this.apiKey = apiKey;
                return this;
            }

            public ProviderConfigBuilder endpoint(String endpoint) {
                this.endpoint = endpoint;
                return this;
            }

            public ProviderConfigBuilder additionalParam(String key, String value) {
                this.additionalParams.put(key, value);
                return this;
            }

            public ProviderConfig build() {
                return new ProviderConfig(this);
            }
        }
    }
}
