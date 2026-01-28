package tech.kayys.golek.core.engine;

import java.util.List;
import java.util.Optional;

import tech.kayys.golek.plugin.api.GolekPlugin;

/**
 * Registry for Golek plugins.
 * Placeholder interface - actual implementation in golek-engine module.
 */
public interface GolekPluginRegistry {

    /**
     * Get all registered plugins
     */
    List<GolekPlugin> all();

    /**
     * Get plugin by ID
     */
    Optional<GolekPlugin> byId(String pluginId);

    /**
     * Get plugins by type
     */
    <T extends GolekPlugin> List<T> byType(Class<T> type);

    /**
     * Get plugin count
     */
    int count();
}
