package tech.kayys.golek.spi.plugin;

import java.util.Optional;

/**
 * Context provided to plugins during initialization.
 * Provides access to engine services and configuration.
 */
public interface PluginContext {

    /**
     * Get the ID of the plugin this context belongs to.
     */
    String getPluginId();

    /**
     * Get a configuration value for this plugin.
     * 
     * @param key The configuration key
     * @return Optional containing the value if present
     */
    Optional<String> getConfig(String key);

    /**
     * Get a configuration value with a default.
     */
    default String getConfig(String key, String defaultValue) {
        return getConfig(key).orElse(defaultValue);
    }
}
