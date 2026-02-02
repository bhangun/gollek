package tech.kayys.golek.engine.plugin;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.golek.api.inference.InferencePhase;
import tech.kayys.golek.core.plugin.InferencePhasePlugin;
import tech.kayys.golek.plugin.api.GolekPlugin;

import org.jboss.logging.Logger;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * Default implementation of plugin registry.
 * Thread-safe and supports concurrent plugin registration.
 */
@ApplicationScoped
public class DefaultPluginRegistry implements PluginRegistry {

    private static final Logger LOG = Logger.getLogger(DefaultPluginRegistry.class);

    private final Map<String, GolekPlugin> plugins = new ConcurrentHashMap<>();
    private final Map<InferencePhase, List<InferencePhasePlugin>> phasePlugins = new ConcurrentHashMap<>();

    @Override
    public void register(GolekPlugin plugin) {
        Objects.requireNonNull(plugin, "plugin cannot be null");

        String id = plugin.id();
        if (plugins.containsKey(id)) {
            LOG.warnf("Plugin %s is already registered, replacing", id);
        }

        plugins.put(id, plugin);
        LOG.infof("Registered plugin: %s (%s)", plugin.name(), id);

        // Index phase plugins
        if (plugin instanceof InferencePhasePlugin phasePlugin) {
            InferencePhase phase = phasePlugin.phase();
            phasePlugins.computeIfAbsent(phase, k -> new ArrayList<>())
                    .add(phasePlugin);

            // Sort by execution order
            phasePlugins.get(phase).sort(
                    Comparator.comparing(Plugin::order));

            LOG.debugf("Registered phase plugin for %s: %s", phase, id);
        }
    }

    @Override
    public void unregister(String pluginId) {
        GolekPlugin removed = plugins.remove(pluginId);
        if (removed != null) {
            LOG.infof("Unregistered plugin: %s", pluginId);

            // Remove from phase index
            if (removed instanceof InferencePhasePlugin phasePlugin) {
                InferencePhase phase = phasePlugin.phase();
                List<InferencePhasePlugin> list = phasePlugins.get(phase);
                if (list != null) {
                    list.remove(phasePlugin);
                }
            }
        }
    }

    @Override
    public Optional<GolekPlugin> getPlugin(String pluginId) {
        return Optional.ofNullable(plugins.get(pluginId));
    }

    @Override
    public List<GolekPlugin> getAllPlugins() {
        return new ArrayList<>(plugins.values());
    }

    @Override
    @SuppressWarnings("unchecked")
    public <T extends GolekPlugin> List<T> getPluginsByType(Class<T> type) {
        return plugins.values().stream()
                .filter(type::isInstance)
                .map(p -> (T) p)
                .collect(Collectors.toList());
    }

    @Override
    public List<InferencePhasePlugin> getPluginsForPhase(InferencePhase phase) {
        return phasePlugins.getOrDefault(phase, Collections.emptyList());
    }

    @Override
    public boolean isRegistered(String pluginId) {
        return plugins.containsKey(pluginId);
    }

    @Override
    public int count() {
        return plugins.size();
    }

    @Override
    public void clear() {
        LOG.warn("Clearing all plugins from registry");
        plugins.clear();
        phasePlugins.clear();
    }

    /**
     * Get plugin statistics
     */
    public PluginStatistics getStatistics() {
        Map<InferencePhase, Integer> phaseCount = phasePlugins.entrySet().stream()
                .collect(Collectors.toMap(
                        Map.Entry::getKey,
                        e -> e.getValue().size()));

        return new PluginStatistics(
                plugins.size(),
                phasePlugins.values().stream().mapToInt(List::size).sum(),
                phaseCount);
    }

    public record PluginStatistics(
            int totalPlugins,
            int phasePlugins,
            Map<InferencePhase, Integer> pluginsPerPhase) {
    }
}