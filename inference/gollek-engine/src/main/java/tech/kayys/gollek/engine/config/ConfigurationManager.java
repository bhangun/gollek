package tech.kayys.gollek.engine.config;

import org.eclipse.microprofile.config.Config;
import org.eclipse.microprofile.config.spi.ConfigProviderResolver;
import org.jboss.logging.Logger;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Central configuration management system with MicroProfile Config integration.
 * 
 * <p>
 * Provides a unified configuration API with fallback chain:
 * </p>
 * <ol>
 * <li>Runtime configuration (dynamic overrides)</li>
 * <li>Plugin-specific configuration</li>
 * <li>Global configuration</li>
 * <li>MicroProfile Config (application.properties)</li>
 * </ol>
 * 
 * <p>
 * Thread-safe and supports configuration change listeners.
 * </p>
 * 
 * @author Bhangun
 */
public class ConfigurationManager {
    private static final Logger LOG = Logger.getLogger(ConfigurationManager.class);

    // Core configuration storage
    private final Map<String, Object> globalConfig = new ConcurrentHashMap<>();
    private final Map<String, Map<String, Object>> pluginConfigs = new ConcurrentHashMap<>();
    private final Map<String, Object> runtimeConfig = new ConcurrentHashMap<>();

    // Listeners for configuration changes
    private final Map<String, ConfigChangeListener> listeners = new ConcurrentHashMap<>();

    // MicroProfile Config integration
    private final Config microProfileConfig;

    /**
     * Create configuration manager with MicroProfile Config integration.
     */
    public ConfigurationManager() {
        Config mpConfig;
        try {
            mpConfig = ConfigProviderResolver.instance().getConfig();
        } catch (Exception e) {
            LOG.warn("MicroProfile Config not available, using in-memory configuration only");
            mpConfig = null;
        }
        this.microProfileConfig = mpConfig;
    }

    // ============================================================================
    // Global Configuration
    // ============================================================================

    /**
     * Set global configuration property.
     * Notifies listeners of the change.
     */
    public void setGlobalProperty(String key, Object value) {
        Object oldValue = globalConfig.put(key, value);
        notifyListeners("global", key, oldValue, value);
    }

    /**
     * Get global configuration property.
     */
    public Object getGlobalProperty(String key) {
        return globalConfig.get(key);
    }

    /**
     * Get global configuration property with type safety and default value.
     */
    public <T> T getGlobalProperty(String key, Class<T> type, T defaultValue) {
        Object value = globalConfig.get(key);
        if (value != null && type.isInstance(value)) {
            return type.cast(value);
        }
        return defaultValue;
    }

    /**
     * Get all global configuration (defensive copy).
     */
    public Map<String, Object> getGlobalConfig() {
        return new ConcurrentHashMap<>(globalConfig);
    }

    // ============================================================================
    // Plugin-Specific Configuration
    // ============================================================================

    /**
     * Set plugin-specific configuration property.
     * Notifies listeners of the change.
     */
    public void setPluginProperty(String pluginId, String key, Object value) {
        pluginConfigs.computeIfAbsent(pluginId, k -> new ConcurrentHashMap<>()).put(key, value);
        notifyListeners(pluginId, key, null, value);
    }

    /**
     * Get plugin-specific configuration property.
     */
    public Object getPluginProperty(String pluginId, String key) {
        Map<String, Object> pluginConfig = pluginConfigs.get(pluginId);
        return pluginConfig != null ? pluginConfig.get(key) : null;
    }

    /**
     * Get plugin-specific configuration property with type safety and default
     * value.
     */
    public <T> T getPluginProperty(String pluginId, String key, Class<T> type, T defaultValue) {
        Object value = getPluginProperty(pluginId, key);
        if (value != null && type.isInstance(value)) {
            return type.cast(value);
        }
        return defaultValue;
    }

    /**
     * Get all plugin configuration (defensive copy).
     */
    public Map<String, Object> getPluginConfig(String pluginId) {
        Map<String, Object> config = pluginConfigs.get(pluginId);
        return config != null ? new ConcurrentHashMap<>(config) : new ConcurrentHashMap<>();
    }

    /**
     * Get merged configuration for a plugin (global + plugin-specific).
     * Plugin-specific values override global values.
     */
    public Map<String, Object> getMergedConfig(String pluginId) {
        Map<String, Object> merged = new HashMap<>(globalConfig);
        Map<String, Object> pluginConfig = pluginConfigs.get(pluginId);
        if (pluginConfig != null) {
            merged.putAll(pluginConfig);
        }
        return merged;
    }

    // ============================================================================
    // Runtime Configuration (Dynamic Overrides)
    // ============================================================================

    /**
     * Set runtime configuration property (highest priority, dynamic).
     * Use for temporary overrides that don't persist.
     */
    public void setRuntimeProperty(String key, Object value) {
        runtimeConfig.put(key, value);
    }

    /**
     * Get runtime configuration property.
     */
    public Object getRuntimeProperty(String key) {
        return runtimeConfig.get(key);
    }

    /**
     * Get all runtime configuration (defensive copy).
     */
    public Map<String, Object> getRuntimeConfig() {
        return new ConcurrentHashMap<>(runtimeConfig);
    }

    /**
     * Clear all runtime configuration.
     */
    public void clearRuntimeConfig() {
        runtimeConfig.clear();
    }

    // ============================================================================
    // Unified Property Access with Fallback Chain
    // ============================================================================

    /**
     * Get configuration property with automatic fallback chain.
     * 
     * <p>
     * Fallback order:
     * </p>
     * <ol>
     * <li>Runtime configuration</li>
     * <li>Plugin-specific configuration (if pluginId provided)</li>
     * <li>Global configuration</li>
     * <li>MicroProfile Config (application.properties)</li>
     * </ol>
     * 
     * @param key          Configuration key
     * @param pluginId     Optional plugin ID for plugin-specific config
     * @param type         Expected type
     * @param defaultValue Default value if not found
     * @return Configuration value or default
     */
    public <T> T getProperty(String key, String pluginId, Class<T> type, T defaultValue) {
        // 1. Check runtime configuration (highest priority)
        Object value = runtimeConfig.get(key);
        if (value != null && type.isInstance(value)) {
            return type.cast(value);
        }

        // 2. Check plugin-specific configuration
        if (pluginId != null) {
            value = getPluginProperty(pluginId, key);
            if (value != null && type.isInstance(value)) {
                return type.cast(value);
            }
        }

        // 3. Check global configuration
        value = getGlobalProperty(key);
        if (value != null && type.isInstance(value)) {
            return type.cast(value);
        }

        // 4. Fallback to MicroProfile Config
        if (microProfileConfig != null) {
            try {
                return getMicroProfileProperty(key, type);
            } catch (Exception e) {
                LOG.debugf("MicroProfile Config does not contain key: %s", key);
            }
        }

        return defaultValue;
    }

    /**
     * Get configuration property without plugin context.
     */
    public <T> T getProperty(String key, Class<T> type, T defaultValue) {
        return getProperty(key, null, type, defaultValue);
    }

    /**
     * Get configuration property as string with default.
     */
    public String getString(String key, String defaultValue) {
        return getProperty(key, null, String.class, defaultValue);
    }

    /**
     * Get configuration property as integer with default.
     */
    public Integer getInteger(String key, Integer defaultValue) {
        return getProperty(key, null, Integer.class, defaultValue);
    }

    /**
     * Get configuration property as long with default.
     */
    public Long getLong(String key, Long defaultValue) {
        return getProperty(key, null, Long.class, defaultValue);
    }

    /**
     * Get configuration property as boolean with default.
     */
    public Boolean getBoolean(String key, Boolean defaultValue) {
        return getProperty(key, null, Boolean.class, defaultValue);
    }

    /**
     * Get configuration property as double with default.
     */
    public Double getDouble(String key, Double defaultValue) {
        return getProperty(key, null, Double.class, defaultValue);
    }

    /**
     * Get configuration property as string for plugin.
     */
    public String getString(String key, String pluginId, String defaultValue) {
        return getProperty(key, pluginId, String.class, defaultValue);
    }

    /**
     * Get configuration property as integer for plugin.
     */
    public Integer getInteger(String key, String pluginId, Integer defaultValue) {
        return getProperty(key, pluginId, Integer.class, defaultValue);
    }

    /**
     * Get configuration property as long for plugin.
     */
    public Long getLong(String key, String pluginId, Long defaultValue) {
        return getProperty(key, pluginId, Long.class, defaultValue);
    }

    /**
     * Get configuration property as boolean for plugin.
     */
    public Boolean getBoolean(String key, String pluginId, Boolean defaultValue) {
        return getProperty(key, pluginId, Boolean.class, defaultValue);
    }

    /**
     * Get configuration property as double for plugin.
     */
    public Double getDouble(String key, String pluginId, Double defaultValue) {
        return getProperty(key, pluginId, Double.class, defaultValue);
    }

    // ============================================================================
    // Listener Support
    // ============================================================================

    /**
     * Add configuration change listener for a plugin.
     */
    public void addConfigChangeListener(String pluginId, ConfigChangeListener listener) {
        listeners.put(pluginId, listener);
    }

    /**
     * Remove configuration change listener for a plugin.
     */
    public void removeConfigChangeListener(String pluginId) {
        listeners.remove(pluginId);
    }

    private void notifyListeners(String pluginId, String key, Object oldValue, Object newValue) {
        ConfigChangeListener listener = listeners.get(pluginId);
        if (listener != null) {
            try {
                listener.onConfigChanged(pluginId, key, oldValue, newValue);
            } catch (Exception e) {
                LOG.errorf(e, "Error notifying config change listener for plugin: %s", pluginId);
            }
        }
    }

    // ============================================================================
    // MicroProfile Config Integration
    // ============================================================================

    /**
     * Get MicroProfile Config value with type conversion.
     */
    @SuppressWarnings("unchecked")
    private <T> T getMicroProfileProperty(String key, Class<T> type) {
        if (microProfileConfig == null) {
            return null;
        }

        if (type == String.class) {
            return (T) microProfileConfig.getValue(key, String.class);
        } else if (type == Integer.class || type == int.class) {
            return (T) microProfileConfig.getValue(key, Integer.class);
        } else if (type == Long.class || type == long.class) {
            return (T) microProfileConfig.getValue(key, Long.class);
        } else if (type == Boolean.class || type == boolean.class) {
            return (T) microProfileConfig.getValue(key, Boolean.class);
        } else if (type == Double.class || type == double.class) {
            return (T) microProfileConfig.getValue(key, Double.class);
        } else if (type == Float.class || type == float.class) {
            return (T) microProfileConfig.getValue(key, Float.class);
        } else if (type == Short.class || type == short.class) {
            return (T) microProfileConfig.getValue(key, Short.class);
        } else if (type == Byte.class || type == byte.class) {
            return (T) microProfileConfig.getValue(key, Byte.class);
        }

        // For other types, try as string and let caller handle conversion
        String value = microProfileConfig.getValue(key, String.class);
        return (T) value;
    }

    /**
     * Get the underlying MicroProfile Config instance.
     */
    public Config getMicroProfileConfig() {
        return microProfileConfig;
    }

    // ============================================================================
    // Configuration Change Listener Interface
    // ============================================================================

    /**
     * Listener for configuration changes.
     */
    @FunctionalInterface
    public interface ConfigChangeListener {
        /**
         * Called when a configuration property changes.
         * 
         * @param pluginId Plugin ID (or "global" for global config)
         * @param key      Configuration key that changed
         * @param oldValue Previous value (null if new)
         * @param newValue New value (null if removed)
         */
        void onConfigChanged(String pluginId, String key, Object oldValue, Object newValue);
    }
}
