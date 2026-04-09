package tech.kayys.gollek.sdk.config;

import tech.kayys.gollek.spi.model.ModelFormat;

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
    private final ModelFormat modelFormat; // v0.1.4
    private final String ggufBasePath; // v0.1.4
    private final String safetensorsBasePath; // v0.1.4
    private final String baseUrl; // v0.1.4

    private SdkConfig(Builder builder) {
        this.apiKey = builder.apiKey;
        this.requestTimeout = builder.requestTimeout;
        this.connectTimeout = builder.connectTimeout;
        this.preferredProvider = builder.preferredProvider;
        this.enableMetrics = builder.enableMetrics;
        this.retryConfig = builder.retryConfig;
        this.modelFormat = builder.modelFormat;
        this.ggufBasePath = builder.ggufBasePath;
        this.safetensorsBasePath = builder.safetensorsBasePath;
        this.baseUrl = builder.baseUrl;
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

    // ── v0.1.4 accessors ──────────────────────────────────────────────────

    /** Explicit model format hint — skip magic-byte detection when set. */
    public Optional<ModelFormat> getModelFormat() {
        return Optional.ofNullable(modelFormat);
    }

    /**
     * Base directory for GGUF models. Defaults to
     * {@code $HOME/.gollek/models/gguf}.
     */
    public String getGgufBasePath() {
        return ggufBasePath != null ? ggufBasePath
                : System.getProperty("user.home") + "/.gollek/models/gguf";
    }

    /**
     * Base directory for SafeTensors checkpoints. Defaults to
     * {@code $HOME/.gollek/models/safetensors}.
     */
    public String getSafetensorsBasePath() {
        return safetensorsBasePath != null ? safetensorsBasePath
                : System.getProperty("user.home") + "/.gollek/models/safetensors";
    }

    /** Base URL for the remote Gollek engine HTTP API. */
    public Optional<String> getBaseUrl() {
        return Optional.ofNullable(baseUrl).filter(s -> !s.isBlank());
    }

    public static final class Builder {
        private String apiKey = "community";
        private Duration requestTimeout = Duration.ofSeconds(60);
        private Duration connectTimeout = Duration.ofSeconds(30);
        private String preferredProvider;
        private boolean enableMetrics;
        private RetryConfig retryConfig = RetryConfig.builder().build();
        private ModelFormat modelFormat; // v0.1.4
        private String ggufBasePath; // v0.1.4
        private String safetensorsBasePath; // v0.1.4
        private String baseUrl; // v0.1.4

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

        /** Hint the format so the SDK can skip filesystem detection. */
        public Builder modelFormat(ModelFormat v) {
            this.modelFormat = v;
            return this;
        }

        /** Override GGUF model base directory. */
        public Builder ggufBasePath(String v) {
            this.ggufBasePath = v;
            return this;
        }

        /** Override SafeTensors base directory. */
        public Builder safetensorsBasePath(String v) {
            this.safetensorsBasePath = v;
            return this;
        }

        /** Base URL for remote SDK instances. */
        public Builder baseUrl(String v) {
            this.baseUrl = v;
            return this;
        }

        public SdkConfig build() {
            return new SdkConfig(this);
        }
    }
}
