package tech.kayys.gollek.spi.plugin;

import io.smallrye.mutiny.Uni;

/**
 * Base interface for all plugins in the inference kernel.
 * Plugins extend the kernel's behavior without modifying core logic.
 */
public interface GollekPlugin {

    /**
     * Unique plugin identifier (namespace/name format)
     * Example: "tech.kayys/validation-plugin"
     */
    String id();

    /**
     * Plugin execution order. Lower numbers execute first.
     * Default: 100
     */
    default int order() {
        return 100;
    }

    /**
     * Plugin version (semantic versioning)
     */
    default String version() {
        return "1.0.0";
    }

    /**
     * Initialize plugin with context.
     * Called once during engine startup.
     */
    default Uni<Void> initialize(PluginContext context) {
        return Uni.createFrom().voidItem();
    }

    /**
     * Start the plugin operations.
     */
    default Uni<Void> start() {
        return Uni.createFrom().voidItem();
    }

    /**
     * Stop the plugin operations.
     */
    default Uni<Void> stop() {
        return Uni.createFrom().voidItem();
    }

    /**
     * Shutdown plugin and release resources.
     */
    default Uni<Void> shutdown() {
        return Uni.createFrom().voidItem();
    }

    /**
     * Update plugin configuration dynamically.
     */
    default void onConfigUpdate(java.util.Map<String, Object> newConfig) {
        // Default: no-op
    }

    /**
     * Get current plugin configuration.
     */
    default java.util.Map<String, Object> currentConfig() {
        return java.util.Collections.emptyMap();
    }

    /**
     * Check if plugin is healthy and operational
     */
    default boolean isHealthy() {
        return true;
    }

    /**
     * Get plugin metadata
     */
    default PluginMetadata metadata() {
        return new PluginMetadata(
                id(),
                version(),
                getClass().getSimpleName(),
                order());
    }

    /**
     * Plugin metadata record
     */
    record PluginMetadata(
            String id,
            String version,
            String implementationClass,
            int order) {
    }
}
