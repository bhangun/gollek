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

package tech.kayys.gollek.plugin.core;

import org.jboss.logging.Logger;
import tech.kayys.gollek.spi.plugin.GollekPlugin;
import tech.kayys.gollek.spi.plugin.PluginContext;

import jakarta.enterprise.inject.Instance;
import jakarta.inject.Inject;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/**
 * Central manager for plugin lifecycle and discovery.
 * 
 * <p>
 * Supports:
 * </p>
 * <ul>
 * <li>CDI-based plugin discovery</li>
 * <li>JAR-based dynamic loading from ~/.gollek/plugins/</li>
 * <li>Hot-reload on JAR changes</li>
 * <li>Plugin isolation via custom ClassLoaders</li>
 * </ul>
 */
@jakarta.enterprise.context.ApplicationScoped
@io.quarkus.arc.DefaultBean
public class PluginManager {

    private static final Logger LOG = Logger.getLogger(PluginManager.class);
    private static final String PLUGIN_DIR_PROPERTY = "gollek.plugin.directory";

    private final Map<String, GollekPlugin> plugins = new ConcurrentHashMap<>();
    private final List<tech.kayys.gollek.plugin.core.PluginListener> listeners = new ArrayList<>();
    private final JarPluginLoader jarLoader;
    private volatile boolean initialized = false;

    @Inject
    Instance<GollekPlugin> discoveredPlugins;

    /**
     * Create plugin manager with JAR loader support.
     */
    public PluginManager() {
        String pluginDir = System.getProperty(PLUGIN_DIR_PROPERTY,
                System.getProperty("user.home") + "/.gollek/plugins");
        this.jarLoader = new JarPluginLoader(java.nio.file.Paths.get(pluginDir));
    }

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

            // Register CDI discovered plugins
            discoveredPlugins.forEach(this::registerPlugin);

            // Load JAR plugins from ~/.gollek/plugins/
            loadJarPlugins();

            // Initialize all plugins
            initializePlugins();

            // Start watching for plugin changes
            jarLoader.startWatching();

            initialized = true;
            LOG.infof("Plugin manager initialized with %d plugins", plugins.size());
        }
    }

    /**
     * Load plugins from JAR files.
     */
    private void loadJarPlugins() {
        List<GollekPlugin> jarPlugins = jarLoader.loadAll();
        for (GollekPlugin plugin : jarPlugins) {
            registerPlugin(plugin);
        }
    }

    /**
     * Initialize all loaded plugins.
     */
    private void initializePlugins() {
        plugins.values().stream()
                .sorted(Comparator.comparingInt(GollekPlugin::order))
                .forEach(plugin -> {
                    try {
                        PluginContext context = new DefaultPluginContext(plugin.id(), this);
                        plugin.initialize(context);
                        LOG.infof("Initialized plugin: %s", plugin.id());
                    } catch (Exception e) {
                        LOG.errorf(e, "Failed to initialize plugin: %s", plugin.id());
                    }
                });
    }

    /**
     * Start all plugins.
     */
    public void start() {
        ensureInitialized();
        LOG.info("Starting plugins");

        plugins.values().stream()
                .sorted(Comparator.comparingInt(GollekPlugin::order))
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
                .sorted(Comparator.comparingInt(GollekPlugin::order).reversed())
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

    public void registerPlugin(GollekPlugin plugin) {
        if (plugins.containsKey(plugin.id())) {
            LOG.warnf("Plugin %s already registered, replacing", plugin.id());
        }

        plugins.put(plugin.id(), plugin);
        LOG.infof("Registered plugin: %s (version: %s)", plugin.id(), plugin.version());
        notifyPluginRegistered(plugin);
    }

    public void unregisterPlugin(String pluginId) {
        GollekPlugin plugin = plugins.remove(pluginId);
        if (plugin != null) {
            LOG.infof("Unregistered plugin: %s", pluginId);
            notifyPluginUnregistered(plugin);
        }
    }

    public Optional<GollekPlugin> getPlugin(String pluginId) {
        return Optional.ofNullable(plugins.get(pluginId));
    }

    public Collection<GollekPlugin> getAllPlugins() {
        return Collections.unmodifiableCollection(plugins.values());
    }

    public <T extends GollekPlugin> List<T> getPluginsByType(Class<T> type) {
        return plugins.values().stream()
                .filter(type::isInstance)
                .map(type::cast)
                .sorted(Comparator.comparingInt(GollekPlugin::order))
                .collect(Collectors.toList());
    }

    public void addPluginListener(tech.kayys.gollek.plugin.core.PluginListener listener) {
        listeners.add(listener);
    }

    public void removePluginListener(tech.kayys.gollek.plugin.core.PluginListener listener) {
        listeners.remove(listener);
    }

    private void ensureInitialized() {
        if (!initialized) {
            initialize();
        }
    }

    private void notifyPluginRegistered(GollekPlugin plugin) {
        listeners.forEach(listener -> listener.onPluginRegistered(plugin));
    }

    private void notifyPluginUnregistered(GollekPlugin plugin) {
        listeners.forEach(listener -> listener.onPluginUnregistered(plugin));
    }

    private void notifyPluginStarted(GollekPlugin plugin) {
        listeners.forEach(listener -> listener.onPluginStarted(plugin));
    }

    private void notifyPluginStopped(GollekPlugin plugin) {
        listeners.forEach(listener -> listener.onPluginStopped(plugin));
    }
}