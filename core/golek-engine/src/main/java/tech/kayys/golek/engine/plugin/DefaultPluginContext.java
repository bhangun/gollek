package tech.kayys.golek.engine.plugin;

import java.util.HashMap;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

import tech.kayys.golek.engine.context.EngineContext;
import tech.kayys.golek.spi.plugin.PluginContext;
import tech.kayys.golek.spi.plugin.PluginRegistry;

/**
 * Default implementation of plugin context.
 */
public class DefaultPluginContext implements PluginContext {

    private final EngineContext engineContext;
    private final PluginRegistry registry;
    private final Map<String, Object> config;
    private final Map<String, Object> sharedData;

    public DefaultPluginContext(
            EngineContext engineContext,
            PluginRegistry registry) {
        this(engineContext, registry, new HashMap<>());
    }

    public DefaultPluginContext(
            EngineContext engineContext,
            PluginRegistry registry,
            Map<String, Object> config) {
        this.engineContext = engineContext;
        this.registry = registry;
        this.config = new HashMap<>(config);
        this.sharedData = new ConcurrentHashMap<>();
    }

    @Override
    public EngineContext engineContext() {
        return engineContext;
    }

    @Override
    public Map<String, Object> config() {
        return new HashMap<>(config);
    }

    @Override
    @SuppressWarnings("unchecked")
    public <T> Optional<T> getConfig(String key, Class<T> type) {
        Object value = config.get(key);
        if (value != null && type.isInstance(value)) {
            return Optional.of((T) value);
        }
        return Optional.empty();
    }

    @Override
    @SuppressWarnings("unchecked")
    public <T> T getConfigOrDefault(String key, T defaultValue) {
        Object value = config.get(key);
        if (value != null && defaultValue.getClass().isInstance(value)) {
            return (T) value;
        }
        return defaultValue;
    }

    @Override
    public boolean isEnabled() {
        return getConfigOrDefault("enabled", true);
    }

    @Override
    public PluginRegistry registry() {
        return registry;
    }

    @Override
    public void putSharedData(String key, Object value) {
        sharedData.put(key, value);
    }

    @Override
    @SuppressWarnings("unchecked")
    public <T> Optional<T> getSharedData(String key, Class<T> type) {
        Object value = sharedData.get(key);
        if (value != null && type.isInstance(value)) {
            return Optional.of((T) value);
        }
        return Optional.empty();
    }

    @Override
    public String toString() {
        return "DefaultPluginContext{" +
                "configKeys=" + config.keySet() +
                ", sharedDataKeys=" + sharedData.keySet() +
                '}';
    }
}