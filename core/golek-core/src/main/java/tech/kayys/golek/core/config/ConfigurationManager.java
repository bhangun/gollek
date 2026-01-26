package tech.kayys.golek.core.config;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Central configuration management system
 */
public class ConfigurationManager {
    private final Map<String, Object> globalConfig = new ConcurrentHashMap<>();
    private final Map<String, Map<String, Object>> pluginConfigs = new ConcurrentHashMap<>();
    private final Map<String, ConfigChangeListener> listeners = new ConcurrentHashMap<>();

    /**
     * Set global configuration property
     */
    public void setGlobalProperty(String key, Object value) {
        Object oldValue = globalConfig.put(key, value);
        notifyListeners("global", key, oldValue, value);
    }

    /**
     * Get global configuration property
     */
    public Object getGlobalProperty(String key) {
        return globalConfig.get(key);
    }

    /**
     * Get global configuration property with default value
     */
    public <T> T getGlobalProperty(String key, Class<T> type, T defaultValue) {
        Object value = globalConfig.get(key);
        if (value != null && type.isInstance(value)) {
            return type.cast(value);
        }
        return defaultValue;
    }

    /**
     * Set plugin-specific configuration property
     */
    public void setPluginProperty(String pluginId, String key, Object value) {
        pluginConfigs.computeIfAbsent(pluginId, k -> new ConcurrentHashMap<>()).put(key, value);
        notifyListeners(pluginId, key, null, value);
    }

    /**
     * Get plugin-specific configuration property
     */
    public Object getPluginProperty(String pluginId, String key) {
        Map<String, Object> pluginConfig = pluginConfigs.get(pluginId);
        if (pluginConfig != null) {
            return pluginConfig.get(key);
        }
        return null;
    }

    /**
     * Get plugin-specific configuration property with default value
     */
    public <T> T getPluginProperty(String pluginId, String key, Class<T> type, T defaultValue) {
        Object value = getPluginProperty(pluginId, key);
        if (value != null && type.isInstance(value)) {
            return type.cast(value);
        }
        return defaultValue;
    }

    /**
     * Get global configuration
     */
    public Map<String, Object> getGlobalConfig() {
        return new ConcurrentHashMap<>(globalConfig);
    }

    /**
     * Get plugin-specific configuration
     */
    public Map<String, Object> getPluginConfig(String pluginId) {
        Map<String, Object> config = pluginConfigs.get(pluginId);
        return config != null ? new ConcurrentHashMap<>(config) : new ConcurrentHashMap<>();
    }

    /**
     * Get all configuration for a plugin (merged with global)
     */
    public Map<String, Object> getMergedConfig(String pluginId) {
        Map<String, Object> merged = new HashMap<>(globalConfig);
        Map<String, Object> pluginConfig = pluginConfigs.get(pluginId);
        if (pluginConfig != null) {
            merged.putAll(pluginConfig);
        }
        return merged;
    }

    /**
     * Add configuration change listener
     */
    public void addConfigChangeListener(String pluginId, ConfigChangeListener listener) {
        listeners.put(pluginId, listener);
    }

    /**
     * Remove configuration change listener
     */
    public void removeConfigChangeListener(String pluginId) {
        listeners.remove(pluginId);
    }

    private void notifyListeners(String pluginId, String key, Object oldValue, Object newValue) {
        ConfigChangeListener listener = listeners.get(pluginId);
        if (listener != null) {
            listener.onConfigChanged(pluginId, key, oldValue, newValue);
        }
    }

    /**
     * Configuration change listener interface
     */
    public interface ConfigChangeListener {
        void onConfigChanged(String pluginId, String key, Object oldValue, Object newValue);
    }
}