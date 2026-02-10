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
 *
 * @author Bhangun
 */

package tech.kayys.golek.core.plugin;

import org.jboss.logging.Logger;
import tech.kayys.golek.spi.plugin.GolekPlugin;
import tech.kayys.golek.spi.plugin.PluginContext;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Instance;
import jakarta.inject.Inject;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * Central manager for plugin lifecycle and discovery.
 */
@ApplicationScoped
public class PluginManager {

    private static final Logger LOG = Logger.getLogger(PluginManager.class);

    private final Map<String, GolekPlugin> plugins = new ConcurrentHashMap<>();
    private final List<PluginListener> listeners = new ArrayList<>();
    private volatile boolean initialized = false;

    @Inject
    Instance<GolekPlugin> discoveredPlugins;

    /**
     * Initialize the plugin manager and all discovered plugins.
     */
    public void initialize() {
        if (initialized) {
            return;
        }

        synchronized (this) {
            if (initialized)
                return;

            LOG.info("Initializing plugin manager");

            // Register discovered plugins
            discoveredPlugins.forEach(this::registerPlugin);

            // Initialize all plugins
            plugins.values().stream()
                    .sorted(Comparator.comparingInt(GolekPlugin::order))
                    .forEach(plugin -> {
                        try {
                            PluginContext context = new DefaultPluginContext(plugin.id(), this);
                            plugin.initialize(context);
                            LOG.infof("Initialized plugin: %s", plugin.id());
                        } catch (Exception e) {
                            LOG.errorf(e, "Failed to initialize plugin: %s", plugin.id());
                        }
                    });

            initialized = true;
        }
    }

    /**
     * Start all plugins.
     */
    public void start() {
        ensureInitialized();
        LOG.info("Starting plugins");

        plugins.values().stream()
                .sorted(Comparator.comparingInt(GolekPlugin::order))
                .forEach(plugin -> {
                    try {
                        plugin.start();
                        LOG.infof("Started plugin: %s", plugin.id());
                        notifyPluginStarted(plugin);
                    } catch (Exception e) {
                        LOG.errorf(e, "Failed to start plugin: %s", plugin.id());
                    }
                });
    }

    /**
     * Stop all plugins.
     */
    public void stop() {
        if (!initialized) {
            return;
        }

        LOG.info("Stopping plugins");

        // Stop in reverse order
        plugins.values().stream()
                .sorted(Comparator.comparingInt(GolekPlugin::order).reversed())
                .forEach(plugin -> {
                    try {
                        plugin.stop();
                        LOG.infof("Stopped plugin: %s", plugin.id());
                        notifyPluginStopped(plugin);
                    } catch (Exception e) {
                        LOG.errorf(e, "Failed to stop plugin: %s", plugin.id());
                    }
                });
    }

    /**
     * Shutdown all plugins.
     */
    public void shutdown() {
        stop();
        LOG.info("Shutting down plugins");

        plugins.values().forEach(plugin -> {
            try {
                plugin.shutdown();
            } catch (Exception e) {
                LOG.errorf(e, "Error shutting down plugin: %s", plugin.id());
            }
        });

        plugins.clear();
        initialized = false;
    }

    public void registerPlugin(GolekPlugin plugin) {
        if (plugins.containsKey(plugin.id())) {
            LOG.warnf("Plugin %s already registered, replacing", plugin.id());
        }

        plugins.put(plugin.id(), plugin);
        LOG.infof("Registered plugin: %s (version: %s)", plugin.id(), plugin.version());
        notifyPluginRegistered(plugin);
    }

    public void unregisterPlugin(String pluginId) {
        GolekPlugin plugin = plugins.remove(pluginId);
        if (plugin != null) {
            LOG.infof("Unregistered plugin: %s", pluginId);
            notifyPluginUnregistered(plugin);
        }
    }

    public Optional<GolekPlugin> getPlugin(String pluginId) {
        return Optional.ofNullable(plugins.get(pluginId));
    }

    public Collection<GolekPlugin> getAllPlugins() {
        return Collections.unmodifiableCollection(plugins.values());
    }

    public <T extends GolekPlugin> List<T> getPluginsByType(Class<T> type) {
        return plugins.values().stream()
                .filter(type::isInstance)
                .map(type::cast)
                .sorted(Comparator.comparingInt(GolekPlugin::order))
                .collect(Collectors.toList());
    }

    public void addPluginListener(PluginListener listener) {
        listeners.add(listener);
    }

    public void removePluginListener(PluginListener listener) {
        listeners.remove(listener);
    }

    private void ensureInitialized() {
        if (!initialized) {
            initialize();
        }
    }

    private void notifyPluginRegistered(GolekPlugin plugin) {
        listeners.forEach(listener -> listener.onPluginRegistered(plugin));
    }

    private void notifyPluginUnregistered(GolekPlugin plugin) {
        listeners.forEach(listener -> listener.onPluginUnregistered(plugin));
    }

    private void notifyPluginStarted(GolekPlugin plugin) {
        listeners.forEach(listener -> listener.onPluginStarted(plugin));
    }

    private void notifyPluginStopped(GolekPlugin plugin) {
        listeners.forEach(listener -> listener.onPluginStopped(plugin));
    }
}