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

package tech.kayys.gollek.plugin.runner;

import org.jboss.logging.Logger;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.ServiceConfigurationError;
import java.util.ServiceLoader;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Manager for runner plugins.
 * 
 * @since 2.1.0
 */
public class RunnerPluginManager {

    private static final Logger LOG = Logger.getLogger(RunnerPluginManager.class);
    private static final RunnerPluginManager INSTANCE = new RunnerPluginManager();

    private final Map<String, RunnerPlugin> plugins = new ConcurrentHashMap<>();
    private final Map<String, RunnerSession> sessions = new ConcurrentHashMap<>();
    private volatile boolean initialized = false;

    private RunnerPluginManager() {
        discoverServiceLoaderPlugins();
    }

    /**
     * Get the singleton instance.
     */
    public static RunnerPluginManager getInstance() {
        return INSTANCE;
    }

    /**
     * Register a runner plugin.
     */
    public void register(RunnerPlugin plugin) {
        if (plugin == null) {
            throw new IllegalArgumentException("Plugin cannot be null");
        }

        LOG.infof("Registering runner plugin: %s (version %s)",
                plugin.id(), plugin.version());
        plugins.put(plugin.id(), plugin);
    }

    /**
     * Unregister a runner plugin.
     */
    public boolean unregister(String pluginId) {
        RunnerPlugin plugin = plugins.remove(pluginId);
        if (plugin != null) {
            LOG.infof("Unregistering runner plugin: %s", pluginId);
            plugin.shutdown();
            return true;
        }
        return false;
    }

    /**
     * Initialize all registered plugins.
     */
    public void initialize(Map<String, Object> config) {
        if (initialized) {
            LOG.warn("Runner plugin manager already initialized");
            return;
        }

        discoverServiceLoaderPlugins();
        LOG.info("Initializing runner plugins");

        for (RunnerPlugin plugin : plugins.values()) {
            try {
                Map<String, Object> pluginConfig = getPluginConfig(config, plugin.id());
                // Wrap the Map in a RunnerContext for the plugin initialize() API
                RunnerContext ctx = RunnerContext.fromMap(pluginConfig);
                plugin.initialize(ctx);
                LOG.infof("Initialized runner plugin: %s", plugin.id());
            } catch (Exception e) {
                LOG.errorf("Failed to initialize runner plugin %s: %s", plugin.id(), e.getMessage());
            }
        }

        initialized = true;
    }

    /**
     * Get all registered plugins.
     */
    public List<RunnerPlugin> getAllPlugins() {
        return new ArrayList<>(plugins.values());
    }

    /**
     * Get available plugins (ready to use).
     */
    public List<RunnerPlugin> getAvailablePlugins() {
        return plugins.values().stream()
                .filter(RunnerPlugin::isAvailable)
                .sorted(Comparator.comparingInt(RunnerPlugin::priority).reversed())
                .toList();
    }

    /**
     * Get a plugin by ID.
     */
    public Optional<RunnerPlugin> getPlugin(String pluginId) {
        return Optional.ofNullable(plugins.get(pluginId));
    }

    /**
     * Find a plugin that supports the given model.
     */
    public Optional<RunnerPlugin> findPluginForModel(String modelPath) {
        return plugins.values().stream()
                .filter(plugin -> plugin.supportsModel(modelPath))
                .filter(RunnerPlugin::isAvailable)
                .max(Comparator.comparingInt(RunnerPlugin::priority));
    }

    /**
     * Create a session for the given model.
     */
    public Optional<RunnerSession> createSession(String modelPath, Map<String, Object> config) {
        return findPluginForModel(modelPath)
                .map(plugin -> {
                    LOG.infof("Creating session with plugin: %s for model: %s",
                            plugin.id(), modelPath);
                    RunnerSession session = plugin.createSession(modelPath, config);
                    sessions.put(session.getSessionId(), session);
                    return session;
                });
    }

    /**
     * Get a session by ID.
     */
    public Optional<RunnerSession> getSession(String sessionId) {
        return Optional.ofNullable(sessions.get(sessionId));
    }

    /**
     * Close a session.
     */
    public boolean closeSession(String sessionId) {
        RunnerSession session = sessions.remove(sessionId);
        if (session != null) {
            session.close();
            return true;
        }
        return false;
    }

    /**
     * Get plugin health status.
     */
    public Map<String, Boolean> getHealthStatus() {
        Map<String, Boolean> status = new HashMap<>();
        for (Map.Entry<String, RunnerPlugin> entry : plugins.entrySet()) {
            status.put(entry.getKey(), entry.getValue().isHealthy());
        }
        return status;
    }

    /**
     * Shutdown all plugins and close all sessions.
     */
    public void shutdown() {
        LOG.info("Shutting down runner plugins");

        // Close all sessions
        for (RunnerSession session : sessions.values()) {
            try {
                session.close();
            } catch (Exception e) {
                LOG.errorf("Error closing session: %s", e.getMessage());
            }
        }
        sessions.clear();

        // Shutdown all plugins
        for (RunnerPlugin plugin : plugins.values()) {
            try {
                plugin.shutdown();
            } catch (Exception e) {
                LOG.errorf("Error shutting down plugin %s: %s", plugin.id(), e.getMessage());
            }
        }

        plugins.clear();
        initialized = false;
    }

    // ───────────────────────────────────────────────────────────────────────
    // Internal Methods
    // ───────────────────────────────────────────────────────────────────────

    private Map<String, Object> getPluginConfig(Map<String, Object> globalConfig, String pluginId) {
        Object pluginConfig = globalConfig.get(pluginId);
        if (pluginConfig instanceof Map) {
            // noinspection unchecked
            return (Map<String, Object>) pluginConfig;
        }
        return Map.of();
    }

    private void discoverServiceLoaderPlugins() {
        ClassLoader loader = Thread.currentThread().getContextClassLoader();
        if (loader == null) {
            loader = RunnerPlugin.class.getClassLoader();
        }

        Iterator<RunnerPlugin> iterator = ServiceLoader.load(RunnerPlugin.class, loader).iterator();
        while (true) {
            RunnerPlugin plugin;
            try {
                if (!iterator.hasNext()) {
                    return;
                }
                plugin = iterator.next();
            } catch (ServiceConfigurationError e) {
                LOG.errorf("Failed to discover runner plugin via ServiceLoader: %s", e.getMessage());
                continue;
            }
            if (plugins.putIfAbsent(plugin.id(), plugin) == null) {
                LOG.infof("Discovered runner plugin: %s (version %s)", plugin.id(), plugin.version());
            }
        }
    }
}
