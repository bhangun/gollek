package tech.kayys.gollek.spi.provider;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.time.Instant;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Provider health status
 */
public final class ProviderHealth {

    public enum Status {
        HEALTHY,
        DEGRADED,
        UNHEALTHY,
        UNKNOWN
    }

    private final Status status;
    private final String message;
    private final Instant timestamp;
    private final Map<String, Object> details;

    @JsonCreator
    public ProviderHealth(
            @JsonProperty("status") Status status,
            @JsonProperty("message") String message,
            @JsonProperty("timestamp") Instant timestamp,
            @JsonProperty("details") Map<String, Object> details) {
        this.status = Objects.requireNonNull(status, "status");
        this.message = message;
        this.timestamp = timestamp != null ? timestamp : Instant.now();
        this.details = details != null
                ? Collections.unmodifiableMap(new HashMap<>(details))
                : Collections.emptyMap();
    }

    public Status status() {
        return status;
    }

    public String message() {
        return message;
    }

    public Instant timestamp() {
        return timestamp;
    }

    public Map<String, Object> details() {
        return details;
    }

    public boolean isHealthy() {
        return status == Status.HEALTHY;
    }

    public boolean isDegraded() {
        return status == Status.DEGRADED;
    }

    public boolean isUnhealthy() {
        return status == Status.UNHEALTHY;
    }

    public static ProviderHealth healthy() {
        return new ProviderHealth(Status.HEALTHY, "Provider is healthy", null, null);
    }

    public static ProviderHealth healthy(String message) {
        return new ProviderHealth(Status.HEALTHY, message, null, null);
    }

    public static ProviderHealth degraded(String message) {
        return new ProviderHealth(Status.DEGRADED, message, null, null);
    }

    public static ProviderHealth unhealthy(String message) {
        return new ProviderHealth(Status.UNHEALTHY, message, null, null);
    }

    public static ProviderHealth unknown() {
        return new ProviderHealth(Status.UNKNOWN, "Health status unknown", null, null);
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private Status status = Status.UNKNOWN;
        private String message;
        private Instant timestamp = Instant.now();
        private final Map<String, Object> details = new HashMap<>();

        public Builder status(Status status) {
            this.status = status;
            return this;
        }

        public Builder message(String message) {
            this.message = message;
            return this;
        }

        public Builder timestamp(Instant timestamp) {
            this.timestamp = timestamp;
            return this;
        }

        public Builder detail(String key, Object value) {
            this.details.put(key, value);
            return this;
        }

        public Builder details(Map<String, Object> details) {
            this.details.putAll(details);
            return this;
        }

        public ProviderHealth build() {
            return new ProviderHealth(status, message, timestamp, details);
        }
    }

    @Override
    public String toString() {
        return "ProviderHealth{" +
                "status=" + status +
                ", message='" + message + '\'' +
                ", timestamp=" + timestamp +
                '}';
    }
}