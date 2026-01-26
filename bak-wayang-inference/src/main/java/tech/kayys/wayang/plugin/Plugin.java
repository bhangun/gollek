package tech.kayys.golek.plugin;

import java.util.Map;

public interface Plugin {
    
    /**
     * Unique identifier for this plugin
     */
    String getId();
    
    /**
     * Human-readable name
     */
    String getName();
    
    /**
     * Plugin version
     */
    String getVersion();
    
    /**
     * Plugin description
     */
    String getDescription();
    
    /**
     * Plugin author
     */
    default String getAuthor() {
        return "Unknown";
    }
    
    /**
     * Minimum required platform version
     */
    default String getRequiredPlatformVersion() {
        return "1.0.0";
    }
    
    /**
     * Dependencies on other plugins (plugin IDs)
     */
    default String[] getDependencies() {
        return new String[0];
    }
    
    /**
     * Initialize the plugin with configuration
     */
    void initialize(PluginContext context) throws PluginException;
    
    /**
     * Start the plugin (called after initialization)
     */
    void start() throws PluginException;
    
    /**
     * Stop the plugin
     */
    void stop() throws PluginException;
    
    /**
     * Get plugin configuration schema
     */
    default Map<String, Object> getConfigSchema() {
        return Map.of();
    }
    
    /**
     * Health check for the plugin
     */
    default boolean isHealthy() {
        return true;
    }
}
