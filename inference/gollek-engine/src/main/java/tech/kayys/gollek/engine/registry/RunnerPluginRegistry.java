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

package tech.kayys.gollek.engine.registry;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Instance;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.gollek.plugin.runner.RunnerPlugin;
import tech.kayys.gollek.plugin.runner.RunnerSession;
import tech.kayys.gollek.plugin.runner.RunnerPluginManager;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Registry for runner plugins.
 * 
 * <p>Integrates the runner plugin system with the Gollek engine,
 * providing automatic discovery, lifecycle management, and session creation.</p>
 * 
 * <h2>Usage</h2>
 * <pre>{@code
 * @Inject
 * RunnerPluginRegistry runnerRegistry;
 * 
 * // Create session for model
 * Optional<RunnerSession> session = runnerRegistry.createSession("model.gguf", config);
 * session.ifPresent(s -> {
 *     InferenceResponse response = s.infer(request).await();
 * });
 * }</pre>
 * 
 * @since 2.1.0
 */
@ApplicationScoped
public class RunnerPluginRegistry {

    private static final Logger LOG = Logger.getLogger(RunnerPluginRegistry.class);

    /**
     * CDI injection point for all available runner plugins.
     */
    @Inject
    Instance<RunnerPlugin> runnerPluginInstances;

    /**
     * Plugin manager singleton.
     * Package-private for testing.
     */
    final RunnerPluginManager pluginManager = RunnerPluginManager.getInstance();

    /**
     * Registry initialization flag.
     * Public for testing.
     */
    public volatile boolean initialized = false;

    /**
     * Initialize the registry by discovering and registering all available runner plugins.
     * 
     * <p>This method is called automatically during engine startup via {@code @PostConstruct}.</p>
     */
    @jakarta.annotation.PostConstruct
    public void discoverRunners() {
        if (initialized) {
            LOG.warn("Runner plugin registry already initialized");
            return;
        }

        LOG.info("Discovering runner plugins...");

        java.util.concurrent.atomic.AtomicInteger registered = new java.util.concurrent.atomic.AtomicInteger(0);
        java.util.concurrent.atomic.AtomicInteger failed = new java.util.concurrent.atomic.AtomicInteger(0);

        if (runnerPluginInstances != null) {
            runnerPluginInstances.stream()
                    .forEach(plugin -> {
                        try {
                            if (plugin.isAvailable()) {
                                pluginManager.register(plugin);
                                LOG.infof("✓ Registered runner plugin: %s (version %s)", 
                                        plugin.id(), plugin.version());
                                registered.incrementAndGet();
                            } else {
                                LOG.debugf("Skipping unavailable plugin: %s", plugin.id());
                            }
                        } catch (Exception e) {
                            LOG.errorf(e, "Failed to register runner plugin: %s", plugin.id());
                            failed.incrementAndGet();
                        }
                    });
        }

        initialized = true;
        LOG.infof("Runner plugin discovery complete. Registered: %d, Failed: %d, Total: %d", 
                  registered.get(), failed.get(), pluginManager.getAllPlugins().size());
    }

    /**
     * Create a runner session for the given model.
     * 
     * <p>Automatically selects the appropriate runner plugin based on the model file extension
     * and creates a session for inference.</p>
     * 
     * @param modelPath Path to the model file
     * @param config Session configuration parameters
     * @return Optional containing the session if a compatible runner is found
     */
    public Optional<RunnerSession> createSession(String modelPath, Map<String, Object> config) {
        if (!initialized) {
            LOG.warn("Runner plugin registry not initialized");
            return Optional.empty();
        }

        LOG.debugf("Creating session for model: %s", modelPath);

        Optional<RunnerSession> session = pluginManager.createSession(modelPath, config);
        
        session.ifPresent(s -> 
            LOG.infof("Created session %s with runner: %s", 
                     s.getSessionId(), s.getRunner().id())
        );

        return session;
    }

    /**
     * Create a runner session with default configuration.
     * 
     * @param modelPath Path to the model file
     * @return Optional containing the session if a compatible runner is found
     */
    public Optional<RunnerSession> createSession(String modelPath) {
        return createSession(modelPath, Map.of());
    }

    /**
     * Initialize all registered runner plugins with configuration.
     * 
     * @param config Global configuration map
     */
    public void initialize(Map<String, Object> config) {
        if (!initialized) {
            LOG.warn("Runner plugin registry not initialized, initializing now");
            discoverRunners();
        }

        LOG.info("Initializing runner plugins with configuration");
        pluginManager.initialize(config);
    }

    /**
     * Get all registered runner plugins.
     * 
     * @return List of all registered plugins
     */
    public List<RunnerPlugin> getAllPlugins() {
        return pluginManager.getAllPlugins();
    }

    /**
     * Get available runner plugins (ready to use).
     * 
     * @return List of available plugins
     */
    public List<RunnerPlugin> getAvailablePlugins() {
        return pluginManager.getAvailablePlugins();
    }

    /**
     * Get a specific runner plugin by ID.
     * 
     * @param pluginId Plugin identifier
     * @return Optional containing the plugin if found
     */
    public Optional<RunnerPlugin> getPlugin(String pluginId) {
        return pluginManager.getPlugin(pluginId);
    }

    /**
     * Find a runner plugin that supports the given model.
     * 
     * @param modelPath Path to the model file
     * @return Optional containing a compatible plugin
     */
    public Optional<RunnerPlugin> findPluginForModel(String modelPath) {
        return pluginManager.findPluginForModel(modelPath);
    }

    /**
     * Get a session by ID.
     * 
     * @param sessionId Session identifier
     * @return Optional containing the session if found
     */
    public Optional<RunnerSession> getSession(String sessionId) {
        return pluginManager.getSession(sessionId);
    }

    /**
     * Close a runner session.
     * 
     * @param sessionId Session identifier
     * @return true if session was closed successfully
     */
    public boolean closeSession(String sessionId) {
        LOG.debugf("Closing session: %s", sessionId);
        return pluginManager.closeSession(sessionId);
    }

    /**
     * Get health status of all runner plugins.
     * 
     * @return Map of plugin IDs to health status
     */
    public Map<String, Boolean> getHealthStatus() {
        return pluginManager.getHealthStatus();
    }

    /**
     * Get registry statistics.
     * 
     * @return Statistics map
     */
    public Map<String, Object> getStats() {
        Map<String, Object> stats = new HashMap<>();
        stats.put("initialized", initialized);
        stats.put("total_plugins", pluginManager.getAllPlugins().size());
        stats.put("available_plugins", pluginManager.getAvailablePlugins().size());
        stats.put("active_sessions", pluginManager.getHealthStatus().size());
        
        // Add plugin details
        List<Map<String, Object>> plugins = pluginManager.getAllPlugins().stream()
                .map(p -> Map.<String, Object>of(
                    "id", p.id(),
                    "name", p.name(),
                    "version", p.version(),
                    "available", p.isAvailable(),
                    "formats", p.supportedFormats()
                ))
                .toList();
        stats.put("plugins", plugins);
        
        return stats;
    }

    /**
     * Shutdown the registry and all runner plugins.
     * 
     * <p>This method is called automatically during engine shutdown.</p>
     */
    @jakarta.annotation.PreDestroy
    public void shutdown() {
        if (!initialized) {
            return;
        }

        LOG.info("Shutting down runner plugin registry");
        pluginManager.shutdown();
        initialized = false;
        LOG.info("Runner plugin registry shutdown complete");
    }
}
