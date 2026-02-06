package tech.kayys.golek.core.plugin;

import tech.kayys.golek.spi.plugin.GolekPlugin;

/**
 * Plugin listener interface
 */
public interface PluginListener {
    /**
     * Called when a plugin is registered
     */
    default void onPluginRegistered(GolekPlugin plugin) {
    }

    /**
     * Called when a plugin is unregistered
     */
    default void onPluginUnregistered(GolekPlugin plugin) {
    }

    /**
     * Called when a plugin is started
     */
    default void onPluginStarted(GolekPlugin plugin) {
    }

    /**
     * Called when a plugin is stopped
     */
    default void onPluginStopped(GolekPlugin plugin) {
    }

    /**
     * Called when a plugin state changes
     */
    // default void onPluginStateChanged(GolekPlugin plugin, PluginState oldState,
    // PluginState newState) {}
}