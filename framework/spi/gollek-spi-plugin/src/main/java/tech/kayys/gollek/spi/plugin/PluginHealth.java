package tech.kayys.gollek.spi.plugin;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/**
 * Represents the health status of a plugin.
 */
public record PluginHealth(
        Status status,
        String message,
        Map<String, Object> details) {

    public enum Status {
        HEALTHY,
        UNHEALTHY,
        DEGRADED,
        UNKNOWN
    }

    public boolean isHealthy() {
        return status == Status.HEALTHY;
    }

    public PluginHealth withDetail(String key, Object value) {
        Map<String, Object> newDetails = new HashMap<>(this.details);
        newDetails.put(key, value);
        return new PluginHealth(this.status, this.message, Collections.unmodifiableMap(newDetails));
    }

    public static PluginHealth healthy() {
        return new PluginHealth(Status.HEALTHY, "Plugin is healthy", Collections.emptyMap());
    }

    public static PluginHealth unhealthy(String message) {
        return new PluginHealth(Status.UNHEALTHY, message, Collections.emptyMap());
    }

    public static PluginHealth unhealthy(String message, Map<String, Object> details) {
        return new PluginHealth(Status.UNHEALTHY, message, details);
    }

    public static PluginHealth degraded(String message, Map<String, Object> details) {
        return new PluginHealth(Status.DEGRADED, message, details);
    }
}
