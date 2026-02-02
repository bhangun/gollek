package tech.kayys.golek.core.plugin;

import tech.kayys.golek.api.plugin.PluginContext;
import java.util.Optional;

/**
 * Default implementation of PluginContext.
 */
public class DefaultPluginContext implements PluginContext {

    private final String pluginId;
    private final PluginManager pluginManager;

    public DefaultPluginContext(String pluginId, PluginManager pluginManager) {
        this.pluginId = pluginId;
        this.pluginManager = pluginManager;
    }

    @Override
    public String getPluginId() {
        return pluginId;
    }

    @Override
    public Optional<String> getConfig(String key) {
        // Placeholder for configuration retrieval logic
        // In a real implementation, this would query a configuration service
        return Optional.empty();
    }
}
