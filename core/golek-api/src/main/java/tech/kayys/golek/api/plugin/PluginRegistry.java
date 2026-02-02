package tech.kayys.golek.api.plugin;

import java.util.List;
import java.util.Optional;

/**
 * Registry for managing plugin lifecycle and discovery.
 */
public interface PluginRegistry {
    void initialize();

    void registerPlugin(GolekPlugin plugin);

    void unregisterPlugin(String pluginId);

    List<GolekPlugin> all();

    <T extends GolekPlugin> List<T> byType(Class<T> type);

    Optional<GolekPlugin> byId(String pluginId);

    void reload(String pluginId);

    List<GolekPlugin.PluginMetadata> listMetadata();

    boolean isHealthy();

    List<String> unhealthyPlugins();

    void shutdownAll();
}
