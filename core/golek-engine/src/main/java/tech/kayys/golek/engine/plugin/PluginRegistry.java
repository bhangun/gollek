package tech.kayys.golek.provider.core.plugin;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Instance;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * Central registry for all plugins.
 * Manages plugin lifecycle, discovery, and access.
 * 
 * Thread-safe and supports hot-reload.
 */
@ApplicationScoped
public class PluginRegistry {

    private static final Logger LOG = Logger.getLogger(PluginRegistry.class);

    @Inject
    Instance<GolekPlugin> pluginInstances;

    // Plugin cache: id -> plugin instance
    private final Map<String, GolekPlugin> pluginCache = new ConcurrentHashMap<>();

    // Initialization flag
    private volatile boolean initialized = false;

    /**
     * Initialize registry and discover plugins
     */
    public void initialize() {
        if (initialized) {
            return;
        }

        synchronized (this) {
            if (initialized) {
                return;
            }

            LOG.info("Initializing plugin registry");

            pluginInstances.stream().forEach(plugin -> {
                try {
                    registerPlugin(plugin);
                } catch (Exception e) {
                    LOG.errorf(e, "Failed to register plugin: %s", plugin.id());
                }
            });

            initialized = true;
            LOG.infof("Plugin registry initialized with %d plugins",
                    pluginCache.size());
        }
    }

    /**
     * Register a plugin instance
     */
    public void registerPlugin(GolekPlugin plugin) {
        Objects.requireNonNull(plugin, "plugin cannot be null");

        String id = plugin.id();
        if (pluginCache.containsKey(id)) {
            LOG.warnf("Plugin %s already registered, replacing", id);
        }

        pluginCache.put(id, plugin);
        LOG.infof("Registered plugin: %s (version: %s, order: %d)",
                id, plugin.version(), plugin.order());
    }

    /**
     * Unregister a plugin
     */
    public void unregisterPlugin(String pluginId) {
        GolekPlugin removed = pluginCache.remove(pluginId);
        if (removed != null) {
            LOG.infof("Unregistered plugin: %s", pluginId);
            try {
                removed.shutdown();
            } catch (Exception e) {
                LOG.errorf(e, "Error shutting down plugin: %s", pluginId);
            }
        }
    }

    /**
     * Get all registered plugins
     */
    public List<GolekPlugin> all() {
        ensureInitialized();
        return new ArrayList<>(pluginCache.values());
    }

    /**
     * Get plugins by type
     */
    @SuppressWarnings("unchecked")
    public <T extends GolekPlugin> List<T> byType(Class<T> type) {
        ensureInitialized();
        return pluginCache.values().stream()
                .filter(type::isInstance)
                .map(p -> (T) p)
                .sorted(Comparator.comparing(GolekPlugin::order))
                .collect(Collectors.toList());
    }

    /**
     * Get plugin by ID
     */
    public Optional<GolekPlugin> byId(String pluginId) {
        ensureInitialized();
        return Optional.ofNullable(pluginCache.get(pluginId));
    }

    /**
     * Reload a specific plugin
     */
    public void reload(String pluginId) {
        LOG.infof("Reloading plugin: %s", pluginId);

        Optional<GolekPlugin> existing = byId(pluginId);
        if (existing.isEmpty()) {
            LOG.warnf("Plugin %s not found for reload", pluginId);
            return;
        }

        GolekPlugin plugin = existing.get();

        try {
            // Shutdown current instance
            plugin.shutdown();

            // Re-initialize
            // Note: In a real implementation, you'd reload from artifact store
            plugin.initialize(null); // Engine context would be provided

            LOG.infof("Plugin %s reloaded successfully", pluginId);
        } catch (Exception e) {
            LOG.errorf(e, "Failed to reload plugin: %s", pluginId);
            throw new PluginReloadException("Failed to reload plugin: " + pluginId, e);
        }
    }

    /**
     * Get plugin metadata for all plugins
     */
    public List<GolekPlugin.PluginMetadata> listMetadata() {
        return all().stream()
                .map(GolekPlugin::metadata)
                .sorted(Comparator.comparing(GolekPlugin.PluginMetadata::order))
                .collect(Collectors.toList());
    }

    /**
     * Check overall plugin health
     */
    public boolean isHealthy() {
        return all().stream().allMatch(GolekPlugin::isHealthy);
    }

    /**
     * Get unhealthy plugins
     */
    public List<String> unhealthyPlugins() {
        return all().stream()
                .filter(p -> !p.isHealthy())
                .map(GolekPlugin::id)
                .collect(Collectors.toList());
    }

    private void ensureInitialized() {
        if (!initialized) {
            initialize();
        }
    }

    /**
     * Shutdown all plugins
     */
    public void shutdownAll() {
        LOG.info("Shutting down all plugins");

        pluginCache.values().forEach(plugin -> {
            try {
                plugin.shutdown();
            } catch (Exception e) {
                LOG.errorf(e, "Error shutting down plugin: %s", plugin.id());
            }
        });

        pluginCache.clear();
        initialized = false;

        LOG.info("All plugins shut down");
    }

    /**
     * Plugin reload exception
     */
    public static class PluginReloadException extends RuntimeException {
        public PluginReloadException(String message, Throwable cause) {
            super(message, cause);
        }
    }
}