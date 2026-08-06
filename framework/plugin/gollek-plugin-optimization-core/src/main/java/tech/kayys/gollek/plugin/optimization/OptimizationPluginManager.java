/*
 * MIT License
 *
 * Copyright (c) 2026 Kayys.tech
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
 */

package tech.kayys.gollek.plugin.optimization;

import org.jboss.logging.Logger;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * Core optimization plugin manager.
 * 
 * @since 2.1.0
 */
public class OptimizationPluginManager {

    private static final Logger LOG = Logger.getLogger(OptimizationPluginManager.class);
    private static final OptimizationPluginManager INSTANCE = new OptimizationPluginManager();

    private final Map<String, OptimizationPlugin> plugins = new ConcurrentHashMap<>();
    private final Map<String, Boolean> availabilityCache = new ConcurrentHashMap<>();
    private volatile boolean initialized = false;

    private OptimizationPluginManager() {
        // Singleton
    }

    /**
     * Get the singleton instance.
     */
    public static OptimizationPluginManager getInstance() {
        return INSTANCE;
    }

    /**
     * Register a plugin.
     */
    public void register(OptimizationPlugin plugin) {
        if (plugin == null) {
            throw new IllegalArgumentException("Plugin cannot be null");
        }

        LOG.infof("Registering optimization plugin: %s (version %s)",
                plugin.id(), plugin.version());
        plugins.put(plugin.id(), plugin);
        availabilityCache.remove(plugin.id());
    }

    /**
     * Unregister a plugin.
     */
    public boolean unregister(String pluginId) {
        OptimizationPlugin plugin = plugins.remove(pluginId);
        if (plugin != null) {
            LOG.infof("Unregistering optimization plugin: %s", pluginId);
            plugin.shutdown();
            availabilityCache.remove(pluginId);
            return true;
        }
        return false;
    }

    /**
     * Initialize all registered plugins.
     */
    public void initialize(Map<String, Object> config) {
        if (initialized) {
            LOG.warn("Plugin manager already initialized");
            return;
        }

        LOG.info("Initializing optimization plugins");

        for (OptimizationPlugin plugin : plugins.values()) {
            try {
                Map<String, Object> pluginConfig = getPluginConfig(config, plugin.id());
                plugin.initialize(pluginConfig);
                LOG.infof("Initialized plugin: %s", plugin.id());
            } catch (Exception e) {
                LOG.errorf("Failed to initialize plugin %s: %s", plugin.id(), e.getMessage());
            }
        }

        initialized = true;
    }

    /**
     * Get all registered plugins.
     */
    public List<OptimizationPlugin> getAllPlugins() {
        return new ArrayList<>(plugins.values());
    }

    /**
     * Get available plugins (hardware-supported).
     */
    public List<OptimizationPlugin> getAvailablePlugins() {
        return plugins.values().stream()
                .filter(this::isPluginAvailable)
                .sorted(Comparator.comparingInt(OptimizationPlugin::priority).reversed())
                .collect(Collectors.toList());
    }

    /**
     * Check if a specific plugin is available.
     */
    public boolean isPluginAvailable(String pluginId) {
        OptimizationPlugin plugin = plugins.get(pluginId);
        if (plugin == null) {
            return false;
        }
        return isPluginAvailable(plugin);
    }

    /**
     * Get a plugin by ID.
     */
    public Optional<OptimizationPlugin> getPlugin(String pluginId) {
        return Optional.ofNullable(plugins.get(pluginId));
    }

    /**
     * Apply all available optimizations.
     */
    public List<String> applyOptimizations(ExecutionContext context) {
        if (!initialized) {
            LOG.warn("Plugin manager not initialized");
            return List.of();
        }

        List<String> applied = new ArrayList<>();

        for (OptimizationPlugin plugin : getAvailablePlugins()) {
            try {
                if (plugin.isAvailable() && plugin.apply(context)) {
                    applied.add(plugin.id());
                    LOG.debugf("Applied optimization: %s", plugin.id());
                }
            } catch (Exception e) {
                LOG.errorf("Failed to apply optimization %s: %s", plugin.id(), e.getMessage());
            }
        }

        return applied;
    }

    /**
     * Get plugin health status.
     */
    public Map<String, Boolean> getHealthStatus() {
        Map<String, Boolean> status = new HashMap<>();
        for (Map.Entry<String, OptimizationPlugin> entry : plugins.entrySet()) {
            status.put(entry.getKey(), entry.getValue().isHealthy());
        }
        return status;
    }

    /**
     * Shutdown all plugins.
     */
    public void shutdown() {
        LOG.info("Shutting down optimization plugins");

        for (OptimizationPlugin plugin : plugins.values()) {
            try {
                plugin.shutdown();
            } catch (Exception e) {
                LOG.errorf("Error shutting down plugin %s: %s", plugin.id(), e.getMessage());
            }
        }

        plugins.clear();
        availabilityCache.clear();
        initialized = false;
    }

    // ───────────────────────────────────────────────────────────────────────
    // Internal Methods
    // ───────────────────────────────────────────────────────────────────────

    private boolean isPluginAvailable(OptimizationPlugin plugin) {
        return availabilityCache.computeIfAbsent(plugin.id(),
                id -> {
                    try {
                        return plugin.isAvailable();
                    } catch (Exception e) {
                        LOG.warnf("Plugin %s availability check failed: %s", id, e.getMessage());
                        return false;
                    }
                });
    }

    private Map<String, Object> getPluginConfig(Map<String, Object> globalConfig, String pluginId) {
        Object pluginConfig = globalConfig.get(pluginId);
        if (pluginConfig instanceof Map) {
            // noinspection unchecked
            return (Map<String, Object>) pluginConfig;
        }
        return Map.of();
    }
}
