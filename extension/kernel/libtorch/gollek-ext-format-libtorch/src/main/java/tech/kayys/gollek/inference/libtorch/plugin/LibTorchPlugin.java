package tech.kayys.gollek.inference.libtorch.plugin;

import tech.kayys.gollek.inference.libtorch.binding.LibTorchBinding;

import java.util.Set;

/**
 * SPI interface for LibTorch plugins.
 * <p>
 * Plugins extend the LibTorch module's capabilities by registering additional
 * operations. They are discovered via {@link java.util.ServiceLoader} and
 * initialized by the {@link LibTorchPluginRegistry}.
 * <p>
 * To create a custom plugin:
 * <ol>
 * <li>Implement this interface</li>
 * <li>Register in
 * {@code META-INF/services/tech.kayys.gollek.inference.libtorch.plugin.LibTorchPlugin}</li>
 * <li>The plugin will be auto-discovered and initialized at startup</li>
 * </ol>
 */
public interface LibTorchPlugin {

    /**
     * Unique plugin identifier.
     */
    String id();

    /**
     * Human-readable plugin name.
     */
    String name();

    /**
     * Plugin priority. Lower values = higher priority.
     * When multiple plugins provide the same operation, the highest-priority
     * plugin wins.
     *
     * @return priority value (default 100)
     */
    default int priority() {
        return 100;
    }

    /**
     * Initialize the plugin with the LibTorch binding layer.
     * Called once during startup after native libraries are loaded.
     *
     * @param binding the LibTorch FFM binding
     */
    void initialize(LibTorchBinding binding);

    /**
     * Get the set of operation IDs this plugin provides.
     * For example: {@code "conv2d"}, {@code "attention"}, {@code "jit_load"}.
     *
     * @return set of operation identifiers
     */
    Set<String> providedOperations();

    /**
     * Check if this plugin's required native symbols are available.
     *
     * @param binding the LibTorch FFM binding to check against
     * @return true if all required symbols are present
     */
    default boolean isAvailable(LibTorchBinding binding) {
        return true;
    }

    /**
     * Shutdown and release plugin resources.
     */
    default void shutdown() {
        // No-op by default
    }
}
