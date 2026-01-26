package tech.kayys.golek.api.model;

import java.util.Map;

/**
 * Health status and diagnostics for a model runner
 */
public record HealthStatus(
        Status status,
        String message,
        Map<String, Object> diagnostics) {

    public enum Status {
        UP,
        DOWN,
        DEGRADED,
        INITIALIZING
    }
    
    public static HealthStatus up(String message) {
        return new HealthStatus(Status.UP, message, Map.of());
    }
    
    public static HealthStatus down(String message) {
        return new HealthStatus(Status.DOWN, message, Map.of());
    }
}
