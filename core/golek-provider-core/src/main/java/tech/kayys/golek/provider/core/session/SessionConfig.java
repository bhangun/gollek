package tech.kayys.golek.provider.core.session;

import java.time.Duration;

/**
 * Configuration for session management
 */
public final class SessionConfig {

    private final int maxConcurrentSessions;
    private final Duration maxIdleTime;
    private final Duration maxAge;
    private final boolean reuseEnabled;
    private final int warmPoolSize;

    private SessionConfig(Builder builder) {
        this.maxConcurrentSessions = builder.maxConcurrentSessions;
        this.maxIdleTime = builder.maxIdleTime;
        this.maxAge = builder.maxAge;
        this.reuseEnabled = builder.reuseEnabled;
        this.warmPoolSize = builder.warmPoolSize;
    }

    public int getMaxConcurrentSessions() {
        return maxConcurrentSessions;
    }

    public Duration getMaxIdleTime() {
        return maxIdleTime;
    }

    public long getMaxIdleTimeMs() {
        return maxIdleTime.toMillis();
    }

    public Duration getMaxAge() {
        return maxAge;
    }

    public long getMaxAgeMs() {
        return maxAge.toMillis();
    }

    public boolean isReuseEnabled() {
        return reuseEnabled;
    }

    public int getWarmPoolSize() {
        return warmPoolSize;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static SessionConfig defaults() {
        return builder().build();
    }

    public static class Builder {
        private int maxConcurrentSessions = 10;
        private Duration maxIdleTime = Duration.ofMinutes(15);
        private Duration maxAge = Duration.ofHours(1);
        private boolean reuseEnabled = true;
        private int warmPoolSize = 2;

        public Builder maxConcurrentSessions(int maxConcurrentSessions) {
            this.maxConcurrentSessions = maxConcurrentSessions;
            return this;
        }

        public Builder maxIdleTime(Duration maxIdleTime) {
            this.maxIdleTime = maxIdleTime;
            return this;
        }

        public Builder maxAge(Duration maxAge) {
            this.maxAge = maxAge;
            return this;
        }

        public Builder reuseEnabled(boolean reuseEnabled) {
            this.reuseEnabled = reuseEnabled;
            return this;
        }

        public Builder warmPoolSize(int warmPoolSize) {
            this.warmPoolSize = warmPoolSize;
            return this;
        }

        public SessionConfig build() {
            return new SessionConfig(this);
        }
    }

    @Override
    public String toString() {
        return "SessionConfig{" +
                "maxConcurrent=" + maxConcurrentSessions +
                ", maxIdle=" + maxIdleTime +
                ", maxAge=" + maxAge +
                ", reuse=" + reuseEnabled +
                ", warmPool=" + warmPoolSize +
                '}';
    }
}