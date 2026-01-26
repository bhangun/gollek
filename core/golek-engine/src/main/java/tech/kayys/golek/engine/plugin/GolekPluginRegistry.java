package tech.kayys.golek.engine.plugin;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.golek.core.plugin.PluginManager;
import tech.kayys.golek.plugin.api.GolekPlugin;

import java.util.List;
import java.util.Optional;

/**
 * Registry facade for plugins.
 * Delegates to PluginManager in core.
 */
@ApplicationScoped
public class GolekPluginRegistry {

    private static final Logger LOG = Logger.getLogger(GolekPluginRegistry.class);

    @Inject
    PluginManager pluginManager;

    /**
     * Initialize registry and discover plugins
     */
    public void initialize() {
        pluginManager.initialize();
    }

    /**
     * Register a plugin instance
     */
    public void registerPlugin(GolekPlugin plugin) {
        pluginManager.registerPlugin(plugin);
    }

    /**
     * Unregister a plugin
     */
    public void unregisterPlugin(String pluginId) {
        pluginManager.unregisterPlugin(pluginId);
    }

    /**
     * Get all registered plugins
     */
    public List<GolekPlugin> all() {
        return (List<GolekPlugin>) pluginManager.getAllPlugins();
    }

    /**
     * Get plugins by type
     */
    public <T extends GolekPlugin> List<T> byType(Class<T> type) {
        return pluginManager.getPluginsByType(type);
    }

    /**
     * Get plugin by ID
     */
    public Optional<GolekPlugin> byId(String pluginId) {
        return pluginManager.getPlugin(pluginId);
    }

    /**
     * Reload a specific plugin
     */
    public void reload(String pluginId) {
        // Reload logic would be delegated to PluginManager if supported
        // For now, we can stop and start if granular reload isn't needed,
        // or just log that it's not fully supported.
        LOG.warnf("Reload not fully implemented in facade for %s", pluginId);
    }

    /**
     * Check overall plugin health
     */
    public boolean isHealthy() {
        return all().stream().allMatch(GolekPlugin::isHealthy);
    }

    /**
     * Shutdown all plugins
     */
    public void shutdownAll() {
        pluginManager.shutdown();
    }
}
