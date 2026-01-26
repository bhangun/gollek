package tech.kayys.golek.core.plugin;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Plugin configuration
 */
public class PluginConfiguration {
    private final Map<String, Object> properties = new ConcurrentHashMap<>();
    
    public void setProperty(String key, Object value) {
        properties.put(key, value);
    }
    
    public Object getProperty(String key) {
        return properties.get(key);
    }
    
    public <T> T getProperty(String key, Class<T> type) {
        Object value = properties.get(key);
        if (value != null && type.isInstance(value)) {
            return type.cast(value);
        }
        return null;
    }
    
    public <T> T getProperty(String key, Class<T> type, T defaultValue) {
        T value = getProperty(key, type);
        return value != null ? value : defaultValue;
    }
    
    public Map<String, Object> getProperties() {
        return new ConcurrentHashMap<>(properties);
    }
    
    public void setProperties(Map<String, Object> newProperties) {
        properties.clear();
        properties.putAll(newProperties);
    }
}