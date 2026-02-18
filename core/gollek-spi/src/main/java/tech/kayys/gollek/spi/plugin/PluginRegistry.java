package tech.kayys.gollek.spi.plugin;

import java.util.List;
import java.util.Optional;

/**
 * Registry for managing plugin lifecycle and discovery.
 */
public interface PluginRegistry {
    void initialize();

    void registerPlugin(GollekPlugin plugin);

    void unregisterPlugin(String pluginId);

    List<GollekPlugin> all();

    <T extends GollekPlugin> List<T> byType(Class<T> type);

    Optional<GollekPlugin> byId(String pluginId);

    void reload(String pluginId);

    List<GollekPlugin.PluginMetadata> listMetadata();

    boolean isHealthy();

    List<String> unhealthyPlugins();

    void shutdownAll();
}
