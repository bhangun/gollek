package tech.kayys.gollek.engine.plugin;

import tech.kayys.gollek.spi.plugin.PluginContext;
import tech.kayys.gollek.spi.plugin.PluginState;

/**
 * Interface for plugin lifecycle management.
 */
public interface PluginLifecycle {

    /**
     * Get the current state of the plugin.
     * 
     * @return PluginState
     */
    PluginState getState();

    /**
     * Initialize the plugin with the given context.
     * 
     * @param context PluginContext
     */
    void initialize(PluginContext context);

    /**
     * Start the plugin.
     */
    void start();

    /**
     * Stop the plugin.
     */
    void stop();

    /**
     * Destroy the plugin (cleanup).
     */
    void destroy();

    /**
     * Check if the plugin is initialized.
     * 
     * @return true if initialized, starting, running, stopping, or stopped.
     */
    boolean isInitialized();

    /**
     * Check if the plugin is running.
     * 
     * @return true if running.
     */
    boolean isRunning();
}
