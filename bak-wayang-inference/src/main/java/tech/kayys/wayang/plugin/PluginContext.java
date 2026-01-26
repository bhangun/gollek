package tech.kayys.golek.plugin;

import java.util.Map;

import tech.kayys.golek.engine.LlamaEngine;

public interface PluginContext {
    
    /**
     * Get the Llama engine instance
     */
    LlamaEngine getEngine();
    
    /**
     * Get plugin configuration
     */
    Map<String, Object> getConfig();
    
    /**
     * Get a configuration value
     */
    <T> T getConfigValue(String key, Class<T> type);
    
    /**
     * Get plugin data directory
     */
    String getDataDirectory();
    
    /**
     * Get shared data between plugins
     */
    Object getSharedData(String key);
    
    /**
     * Set shared data for other plugins
     */
    void setSharedData(String key, Object value);
    
    /**
     * Get another plugin by ID
     */
    <T extends Plugin> T getPlugin(String pluginId, Class<T> type);
    
    /**
     * Register an event listener
     */
    void addEventListener(String eventType, PluginEventListener listener);
    
    /**
     * Emit an event
     */
    void emitEvent(String eventType, Object data);
}
