package tech.kayys.golek.core.config;

import org.eclipse.microprofile.config.Config;
import org.eclipse.microprofile.config.spi.ConfigProviderResolver;
import org.jboss.logging.Logger;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Enhanced configuration system that combines MicroProfile Config with custom
 * configuration
 */
public class EnhancedConfigurationManager {
    private static final Logger LOG = Logger.getLogger(EnhancedConfigurationManager.class);

    private final ConfigurationManager coreConfigManager;
    private final Config microProfileConfig;
    private final Map<String, Object> runtimeConfig = new ConcurrentHashMap<>();

    public EnhancedConfigurationManager(ConfigurationManager coreConfigManager) {
        this.coreConfigManager = coreConfigManager;
        this.microProfileConfig = ConfigProviderResolver.instance().getConfig();
    }

    /**
     * Get configuration property with fallback chain:
     * 1. Runtime configuration
     * 2. Plugin-specific configuration
     * 3. Global configuration
     * 4. MicroProfile Config
     */
    public <T> T getProperty(String key, String pluginId, Class<T> type, T defaultValue) {
        // 1. Check runtime configuration first
        Object value = runtimeConfig.get(key);
        if (value != null && type.isInstance(value)) {
            return type.cast(value);
        }

        // 2. Check plugin-specific configuration
        if (pluginId != null) {
            value = coreConfigManager.getPluginProperty(pluginId, key);
            if (value != null && type.isInstance(value)) {
                return type.cast(value);
            }
        }

        // 3. Check global configuration
        value = coreConfigManager.getGlobalProperty(key);
        if (value != null && type.isInstance(value)) {
            return type.cast(value);
        }

        // 4. Fallback to MicroProfile Config
        try {
            if (type == String.class) {
                return type.cast(microProfileConfig.getValue(key, String.class));
            } else if (type == Integer.class || type == int.class) {
                return type.cast(microProfileConfig.getValue(key, Integer.class));
            } else if (type == Long.class || type == long.class) {
                return type.cast(microProfileConfig.getValue(key, Long.class));
            } else if (type == Boolean.class || type == boolean.class) {
                return type.cast(microProfileConfig.getValue(key, Boolean.class));
            } else if (type == Double.class || type == double.class) {
                return type.cast(microProfileConfig.getValue(key, Double.class));
            }
        } catch (Exception e) {
            LOG.debugf("MicroProfile Config does not contain key: %s", key);
        }

        return defaultValue;
    }

    /**
     * Set runtime configuration property
     */
    public void setRuntimeProperty(String key, Object value) {
        runtimeConfig.put(key, value);
    }

    /**
     * Get runtime configuration property
     */
    public Object getRuntimeProperty(String key) {
        return runtimeConfig.get(key);
    }

    /**
     * Get runtime configuration
     */
    public Map<String, Object> getRuntimeConfig() {
        return new ConcurrentHashMap<>(runtimeConfig);
    }

    /**
     * Clear runtime configuration
     */
    public void clearRuntimeConfig() {
        runtimeConfig.clear();
    }

    /**
     * Get the core configuration manager
     */
    public ConfigurationManager getCoreConfigManager() {
        return coreConfigManager;
    }

    /**
     * Get MicroProfile Config instance
     */
    public Config getMicroProfileConfig() {
        return microProfileConfig;
    }
}