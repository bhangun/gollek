package tech.kayys.gollek.spi.plugin;

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
    default void initialize(PluginContext context) {
        // Default: no-op
    }

    /**
     * Start the plugin operations.
     */
    default void start() {
        // Default: no-op
    }

    /**
     * Stop the plugin operations.
     */
    default void stop() {
        // Default: no-op
    }

    /**
     * Shutdown plugin and release resources.
     */
    default void shutdown() {
        // Default: no-op
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
