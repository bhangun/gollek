package tech.kayys.golek.core.engine;

import java.time.Instant;
import java.util.Map;

/**
 * Health status of the engine or component.
 */
public record HealthStatus(
        Status status,
        String message,
        Instant timestamp,
        Map<String, Object> details) {
    public enum Status {
        HEALTHY,
        DEGRADED,
        UNHEALTHY,
        UNKNOWN
    }

    public static HealthStatus healthy() {
        return new HealthStatus(Status.HEALTHY, "All systems operational", Instant.now(), Map.of());
    }

    public static HealthStatus healthy(String message) {
        return new HealthStatus(Status.HEALTHY, message, Instant.now(), Map.of());
    }

    public static HealthStatus degraded(String message) {
        return new HealthStatus(Status.DEGRADED, message, Instant.now(), Map.of());
    }

    public static HealthStatus unhealthy(String message) {
        return new HealthStatus(Status.UNHEALTHY, message, Instant.now(), Map.of());
    }

    public static HealthStatus unknown() {
        return new HealthStatus(Status.UNKNOWN, "Status unknown", Instant.now(), Map.of());
    }

    public boolean isHealthy() {
        return status == Status.HEALTHY;
    }

    public String getStatus() {
        return status.name();
    }

    public String getMessage() {
        return message;
    }
}
