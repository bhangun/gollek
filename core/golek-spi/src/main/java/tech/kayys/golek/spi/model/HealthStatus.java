package tech.kayys.golek.spi.model;

import java.time.Instant;
import java.util.Map;

/**
 * Health status and diagnostics for a model runner or engine component.
 * Supports both UP/DOWN and HEALTHY/UNHEALTHY status patterns.
 */
public record HealthStatus(
        Status status,
        String message,
        Instant timestamp,
        Map<String, Object> diagnostics) {

    public enum Status {
        UP, // Alias for HEALTHY
        HEALTHY, // Fully operational
        DOWN, // Alias for UNHEALTHY
        UNHEALTHY, // Not operational
        DEGRADED, // Partially operational
        INITIALIZING, // Starting up
        UNKNOWN // Status unknown
    }

    /**
     * Create a health status with current timestamp.
     */
    public HealthStatus(Status status, String message, Map<String, Object> diagnostics) {
        this(status, message, Instant.now(), diagnostics);
    }

    // UP/DOWN pattern (common in Spring Boot actuator)
    public static HealthStatus up(String message) {
        return new HealthStatus(Status.UP, message, Map.of());
    }

    public static HealthStatus down(String message) {
        return new HealthStatus(Status.DOWN, message, Map.of());
    }

    // HEALTHY/UNHEALTHY pattern (common in general health checks)
    public static HealthStatus healthy() {
        return new HealthStatus(Status.HEALTHY, "All systems operational", Map.of());
    }

    public static HealthStatus healthy(String message) {
        return new HealthStatus(Status.HEALTHY, message, Map.of());
    }

    public static HealthStatus degraded(String message) {
        return new HealthStatus(Status.DEGRADED, message, Map.of());
    }

    public static HealthStatus unhealthy(String message) {
        return new HealthStatus(Status.UNHEALTHY, message, Map.of());
    }

    public static HealthStatus unknown() {
        return new HealthStatus(Status.UNKNOWN, "Status unknown", Map.of());
    }

    public static HealthStatus initializing(String message) {
        return new HealthStatus(Status.INITIALIZING, message, Map.of());
    }

    // Convenience check methods
    public boolean isHealthy() {
        return status == Status.HEALTHY || status == Status.UP;
    }

    public boolean isUnhealthy() {
        return status == Status.UNHEALTHY || status == Status.DOWN;
    }

    public boolean isDegraded() {
        return status == Status.DEGRADED;
    }

    // Getters for compatibility
    public String getStatus() {
        return status.name();
    }

    public String getMessage() {
        return message;
    }

    public Map<String, Object> getDiagnostics() {
        return diagnostics;
    }
}
