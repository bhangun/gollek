package tech.kayys.golek.sdk.core.config;

import java.time.Duration;

/**
 * Configuration for retry behavior in the SDK.
 */
public class RetryConfig {
    private final int maxAttempts;
    private final Duration initialBackoff;
    private final Duration maxBackoff;
    private final double backoffMultiplier;
    private final boolean enableJitter;

    private RetryConfig(Builder builder) {
        this.maxAttempts = builder.maxAttempts;
        this.initialBackoff = builder.initialBackoff;
        this.maxBackoff = builder.maxBackoff;
        this.backoffMultiplier = builder.backoffMultiplier;
        this.enableJitter = builder.enableJitter;
    }

    public int getMaxAttempts() {
        return maxAttempts;
    }

    public Duration getInitialBackoff() {
        return initialBackoff;
    }

    public Duration getMaxBackoff() {
        return maxBackoff;
    }

    public double getBackoffMultiplier() {
        return backoffMultiplier;
    }

    public boolean isEnableJitter() {
        return enableJitter;
    }

    /**
     * Calculate backoff duration for a given attempt.
     */
    public Duration calculateBackoff(int attempt) {
        long backoffMillis = (long) (initialBackoff.toMillis() * Math.pow(backoffMultiplier, attempt - 1));
        backoffMillis = Math.min(backoffMillis, maxBackoff.toMillis());
        
        if (enableJitter) {
            // Add random jitter (0-25% of backoff)
            long jitter = (long) (backoffMillis * Math.random() * 0.25);
            backoffMillis += jitter;
        }
        
        return Duration.ofMillis(backoffMillis);
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private int maxAttempts = 3;
        private Duration initialBackoff = Duration.ofMillis(100);
        private Duration maxBackoff = Duration.ofSeconds(30);
        private double backoffMultiplier = 2.0;
        private boolean enableJitter = true;

        public Builder maxAttempts(int maxAttempts) {
            if (maxAttempts < 1) {
                throw new IllegalArgumentException("maxAttempts must be at least 1");
            }
            this.maxAttempts = maxAttempts;
            return this;
        }

        public Builder initialBackoff(Duration initialBackoff) {
            this.initialBackoff = initialBackoff;
            return this;
        }

        public Builder maxBackoff(Duration maxBackoff) {
            this.maxBackoff = maxBackoff;
            return this;
        }

        public Builder backoffMultiplier(double backoffMultiplier) {
            this.backoffMultiplier = backoffMultiplier;
            return this;
        }

        public Builder enableJitter(boolean enableJitter) {
            this.enableJitter = enableJitter;
            return this;
        }

        public RetryConfig build() {
            return new RetryConfig(this);
        }
    }
}
