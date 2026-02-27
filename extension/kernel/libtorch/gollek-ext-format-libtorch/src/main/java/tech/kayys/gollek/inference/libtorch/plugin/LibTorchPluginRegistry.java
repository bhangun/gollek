package tech.kayys.gollek.inference.libtorch.plugin;

import jakarta.enterprise.context.ApplicationScoped;
import org.jboss.logging.Logger;
import tech.kayys.gollek.inference.libtorch.binding.LibTorchBinding;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Registry for LibTorch plugins.
 * <p>
 * Discovers plugins via {@link ServiceLoader}, sorts by priority,
 * and provides lookup by operation ID.
 */
@ApplicationScoped
public class LibTorchPluginRegistry {

    private static final Logger log = Logger.getLogger(LibTorchPluginRegistry.class);

    private final List<LibTorchPlugin> plugins = new ArrayList<>();
    private final ConcurrentHashMap<String, LibTorchPlugin> operationIndex = new ConcurrentHashMap<>();
    private boolean initialized = false;

    /**
     * Discover and initialize all plugins.
     *
     * @param binding the LibTorch FFM binding
     */
    public void initialize(LibTorchBinding binding) {
        if (initialized) {
            log.debug("LibTorchPluginRegistry already initialized");
            return;
        }

        // Discover plugins via ServiceLoader
        ServiceLoader<LibTorchPlugin> loader = ServiceLoader.load(LibTorchPlugin.class);
        List<LibTorchPlugin> discovered = new ArrayList<>();
        for (LibTorchPlugin plugin : loader) {
            discovered.add(plugin);
        }

        // Sort by priority (lower = higher priority)
        discovered.sort(Comparator.comparingInt(LibTorchPlugin::priority));

        // Initialize each plugin
        for (LibTorchPlugin plugin : discovered) {
            try {
                if (plugin.isAvailable(binding)) {
                    plugin.initialize(binding);
                    plugins.add(plugin);

                    // Index operations (first plugin wins for each operation)
                    for (String op : plugin.providedOperations()) {
                        operationIndex.putIfAbsent(op, plugin);
                    }

                    log.infof("Loaded LibTorch plugin: %s (id=%s, priority=%d, ops=%s)",
                            plugin.name(), plugin.id(), plugin.priority(), plugin.providedOperations());
                } else {
                    log.infof("Skipping unavailable plugin: %s (id=%s)", plugin.name(), plugin.id());
                }
            } catch (Exception e) {
                log.warnf(e, "Failed to initialize plugin: %s", plugin.id());
            }
        }

        initialized = true;
        log.infof("LibTorch plugin registry initialized with %d plugins, %d operations",
                plugins.size(), operationIndex.size());
    }

    /**
     * Get the plugin that provides a specific operation.
     *
     * @param operationId the operation identifier
     * @return optional plugin, empty if no plugin provides this operation
     */
    public Optional<LibTorchPlugin> getPlugin(String operationId) {
        return Optional.ofNullable(operationIndex.get(operationId));
    }

    /**
     * Check if an operation is available via any plugin.
     */
    public boolean hasOperation(String operationId) {
        return operationIndex.containsKey(operationId);
    }

    /**
     * Get all registered plugins.
     */
    public List<LibTorchPlugin> getPlugins() {
        return Collections.unmodifiableList(plugins);
    }

    /**
     * Get all available operation IDs.
     */
    public Set<String> getAvailableOperations() {
        return Collections.unmodifiableSet(operationIndex.keySet());
    }

    /**
     * Shutdown all plugins.
     */
    public void shutdown() {
        for (LibTorchPlugin plugin : plugins) {
            try {
                plugin.shutdown();
                log.debugf("Shutdown plugin: %s", plugin.id());
            } catch (Exception e) {
                log.warnf(e, "Error shutting down plugin: %s", plugin.id());
            }
        }
        plugins.clear();
        operationIndex.clear();
        initialized = false;
    }

    public boolean isInitialized() {
        return initialized;
    }
}
