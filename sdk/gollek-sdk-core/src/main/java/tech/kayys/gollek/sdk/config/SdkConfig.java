package tech.kayys.gollek.sdk.config;

import java.time.Duration;
import java.util.Optional;

/**
 * SDK runtime configuration shared by local and remote Gollek SDK builders.
 */
public final class SdkConfig {

    private final String apiKey;
    private final Duration requestTimeout;
    private final Duration connectTimeout;
    private final String preferredProvider;
    private final boolean enableMetrics;
    private final RetryConfig retryConfig;

    private SdkConfig(Builder builder) {
        this.apiKey = builder.apiKey;
        this.requestTimeout = builder.requestTimeout;
        this.connectTimeout = builder.connectTimeout;
        this.preferredProvider = builder.preferredProvider;
        this.enableMetrics = builder.enableMetrics;
        this.retryConfig = builder.retryConfig;
    }

    public static Builder builder() {
        return new Builder();
    }

    public String getApiKey() {
        return apiKey;
    }

    public Duration getRequestTimeout() {
        return requestTimeout;
    }

    public Duration getConnectTimeout() {
        return connectTimeout;
    }

    public Optional<String> getPreferredProvider() {
        return Optional.ofNullable(preferredProvider).filter(v -> !v.isBlank());
    }

    public boolean isEnableMetrics() {
        return enableMetrics;
    }

    public RetryConfig getRetryConfig() {
        return retryConfig;
    }

    public static final class Builder {
        private String apiKey = "community";
        private Duration requestTimeout = Duration.ofSeconds(60);
        private Duration connectTimeout = Duration.ofSeconds(30);
        private String preferredProvider;
        private boolean enableMetrics;
        private RetryConfig retryConfig = RetryConfig.builder().build();

        private Builder() {
        }

        public Builder apiKey(String apiKey) {
            if (apiKey != null && !apiKey.isBlank()) {
                this.apiKey = apiKey;
            }
            return this;
        }

        public Builder requestTimeout(Duration requestTimeout) {
            if (requestTimeout != null) {
                this.requestTimeout = requestTimeout;
            }
            return this;
        }

        public Builder connectTimeout(Duration connectTimeout) {
            if (connectTimeout != null) {
                this.connectTimeout = connectTimeout;
            }
            return this;
        }

        public Builder preferredProvider(String preferredProvider) {
            this.preferredProvider = preferredProvider;
            return this;
        }

        public Builder enableMetrics(boolean enableMetrics) {
            this.enableMetrics = enableMetrics;
            return this;
        }

        public Builder retryConfig(RetryConfig retryConfig) {
            if (retryConfig != null) {
                this.retryConfig = retryConfig;
            }
            return this;
        }

        public Builder maxRetries(int maxAttempts) {
            this.retryConfig = RetryConfig.builder().maxAttempts(maxAttempts).build();
            return this;
        }

        public SdkConfig build() {
            return new SdkConfig(this);
        }
    }
}
