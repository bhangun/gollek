package tech.kayys.golek.engine.plugin;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Instance;
import jakarta.inject.Inject;
import tech.kayys.golek.spi.plugin.PluginRegistry;
import tech.kayys.golek.spi.plugin.GolekPlugin;

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
 * Concrete implementation of the PluginRegistry.
 */
@ApplicationScoped
public class GolekPluginRegistry implements PluginRegistry {

    private static final Logger LOG = Logger.getLogger(GolekPluginRegistry.class);

    @Inject
    Instance<GolekPlugin> pluginInstances;

    // Plugin cache: id -> plugin instance
    private final Map<String, GolekPlugin> pluginCache = new ConcurrentHashMap<>();

    // Initialization flag
    private volatile boolean initialized = false;

    @Override
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

    @Override
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

    @Override
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

    @Override
    public List<GolekPlugin> all() {
        ensureInitialized();
        return new ArrayList<>(pluginCache.values());
    }

    @Override
    @SuppressWarnings("unchecked")
    public <T extends GolekPlugin> List<T> byType(Class<T> type) {
        ensureInitialized();
        return pluginCache.values().stream()
                .filter(type::isInstance)
                .map(p -> (T) p)
                .sorted(Comparator.comparing(GolekPlugin::order))
                .collect(Collectors.toList());
    }

    @Override
    public Optional<GolekPlugin> byId(String pluginId) {
        ensureInitialized();
        return Optional.ofNullable(pluginCache.get(pluginId));
    }

    @Override
    public void reload(String pluginId) {
        LOG.infof("Reloading plugin: %s", pluginId);

        Optional<GolekPlugin> existing = byId(pluginId);
        if (existing.isEmpty()) {
            LOG.warnf("Plugin %s not found for reload", pluginId);
            return;
        }

        GolekPlugin plugin = existing.get();

        try {
            plugin.shutdown();
            plugin.initialize(null);
            LOG.infof("Plugin %s reloaded successfully", pluginId);
        } catch (Exception e) {
            LOG.errorf(e, "Failed to reload plugin: %s", pluginId);
            throw new RuntimeException("Failed to reload plugin: " + pluginId, e);
        }
    }

    @Override
    public List<GolekPlugin.PluginMetadata> listMetadata() {
        return all().stream()
                .map(GolekPlugin::metadata)
                .sorted(Comparator.comparing(GolekPlugin.PluginMetadata::order))
                .collect(Collectors.toList());
    }

    @Override
    public boolean isHealthy() {
        return all().stream().allMatch(GolekPlugin::isHealthy);
    }

    @Override
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

    @Override
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
}
