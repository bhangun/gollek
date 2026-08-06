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

package tech.kayys.gollek.plugin.runner.safetensor.manager;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Instance;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.gollek.plugin.runner.safetensor.feature.SafetensorFeaturePlugin;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Manager for Safetensor feature plugins.
 * 
 * <p>
 * Manages the lifecycle and execution of feature plugins, providing:
 * <ul>
 * <li>Feature plugin discovery and registration</li>
 * <li>Feature enable/disable control</li>
 * <li>Request routing to appropriate features</li>
 * <li>Cross-feature coordination</li>
 * </ul>
 * 
 * <h2>Usage</h2>
 * 
 * <pre>{@code
 * @Inject
 * FeaturePluginManager featureManager;
 * 
 * // Initialize with configuration
 * featureManager.initialize(config);
 * 
 * // Process through features
 * Object result = featureManager.process("audio-feature", audioData);
 * }</pre>
 * 
 * @since 2.1.0
 */
@ApplicationScoped
public class FeaturePluginManager {

    private static final Logger LOG = Logger.getLogger(FeaturePluginManager.class);

    /**
     * CDI injection point for all available feature plugins.
     */
    @Inject
    Instance<SafetensorFeaturePlugin> featurePluginInstances;

    /**
     * Registered feature plugins.
     */
    private final Map<String, SafetensorFeaturePlugin> features = new ConcurrentHashMap<>();

    /**
     * Feature enable/disable state.
     */
    private final Map<String, Boolean> featureStates = new ConcurrentHashMap<>();

    /**
     * Initialization state.
     */
    private volatile boolean initialized = false;

    /**
     * Initialize the feature plugin manager by discovering and registering all
     * available features.
     */
    @jakarta.annotation.PostConstruct
    public void initialize() {
        if (initialized) {
            LOG.warn("Feature plugin manager already initialized");
            return;
        }

        LOG.info("Discovering Safetensor feature plugins...");

        int registered = 0;
        int failed = 0;

        if (featurePluginInstances != null) {
            java.util.concurrent.atomic.AtomicInteger regCount = new java.util.concurrent.atomic.AtomicInteger();
            java.util.concurrent.atomic.AtomicInteger failCount = new java.util.concurrent.atomic.AtomicInteger();
            featurePluginInstances.stream()
                    .forEach(feature -> {
                        try {
                            if (feature.isAvailable()) {
                                register(feature);
                                LOG.infof("✓ Registered feature plugin: %s (version %s)",
                                        feature.id(), feature.version());
                                regCount.incrementAndGet();
                            } else {
                                LOG.debugf("Skipping unavailable feature: %s", feature.id());
                            }
                        } catch (Exception e) {
                            LOG.errorf(e, "Failed to register feature plugin: %s", feature.id());
                            failCount.incrementAndGet();
                        }
                    });
            registered = regCount.get();
            failed = failCount.get();
        }

        initialized = true;
        LOG.infof("Feature plugin discovery complete. Registered: %d, Failed: %d, Total: %d",
                registered, failed, features.size());
    }

    /**
     * Initialize with configuration.
     * 
     * @param config Configuration map
     */
    public void initialize(Map<String, Object> config) {
        initialize();

        // Apply configuration
        if (config.containsKey("features")) {
            @SuppressWarnings("unchecked")
            Map<String, Object> featuresConfig = (Map<String, Object>) config.get("features");
            featuresConfig.forEach((featureId, featureConfig) -> {
                if (featureConfig instanceof Map) {
                    @SuppressWarnings("unchecked")
                    Map<String, Object> fc = (Map<String, Object>) featureConfig;
                    if (fc.containsKey("enabled")) {
                        setFeatureEnabled(featureId, Boolean.parseBoolean(fc.get("enabled").toString()));
                    }
                }
            });
        }
    }

    /**
     * Register a feature plugin.
     * 
     * @param feature Feature plugin to register
     */
    public void register(SafetensorFeaturePlugin feature) {
        if (feature == null) {
            throw new IllegalArgumentException("Feature cannot be null");
        }

        LOG.infof("Registering feature plugin: %s", feature.id());
        features.put(feature.id(), feature);
        featureStates.put(feature.id(), true); // Enabled by default

        // Initialize feature with empty config
        feature.initialize(Map.of());
    }

    /**
     * Unregister a feature plugin.
     * 
     * @param featureId Feature ID to unregister
     * @return true if feature was unregistered
     */
    public boolean unregister(String featureId) {
        SafetensorFeaturePlugin feature = features.remove(featureId);
        if (feature != null) {
            LOG.infof("Unregistering feature plugin: %s", featureId);
            feature.shutdown();
            featureStates.remove(featureId);
            return true;
        }
        return false;
    }

    /**
     * Enable or disable a feature.
     * 
     * @param featureId Feature ID
     * @param enabled   Enable state
     */
    public void setFeatureEnabled(String featureId, boolean enabled) {
        if (features.containsKey(featureId)) {
            featureStates.put(featureId, enabled);
            LOG.infof("Feature %s %s", featureId, enabled ? "enabled" : "disabled");
        } else {
            LOG.warnf("Unknown feature: %s", featureId);
        }
    }

    /**
     * Check if a feature is enabled.
     * 
     * @param featureId Feature ID
     * @return true if feature is enabled
     */
    public boolean isFeatureEnabled(String featureId) {
        return features.containsKey(featureId) &&
                featureStates.getOrDefault(featureId, true);
    }

    /**
     * Get a feature plugin by ID.
     * 
     * @param featureId Feature ID
     * @return Optional containing the feature if found
     */
    public Optional<SafetensorFeaturePlugin> getFeature(String featureId) {
        return Optional.ofNullable(features.get(featureId));
    }

    /**
     * Get all registered features.
     * 
     * @return List of all registered features
     */
    public List<SafetensorFeaturePlugin> getAllFeatures() {
        return new ArrayList<>(features.values());
    }

    /**
     * Get all available (enabled) features.
     * 
     * @return List of available features
     */
    public List<SafetensorFeaturePlugin> getAvailableFeatures() {
        return features.values().stream()
                .filter(f -> isFeatureEnabled(f.id()))
                .filter(SafetensorFeaturePlugin::isAvailable)
                .sorted(Comparator.comparingInt(SafetensorFeaturePlugin::priority).reversed())
                .toList();
    }

    /**
     * Find a feature that supports the given model.
     * 
     * @param modelId Model ID
     * @return Optional containing a compatible feature
     */
    public Optional<SafetensorFeaturePlugin> findFeatureForModel(String modelId) {
        return getAvailableFeatures().stream()
                .filter(f -> f.supportedModels().contains(modelId.toLowerCase()))
                .findFirst();
    }

    /**
     * Find features that support the given input type.
     * 
     * @param inputType Input type (e.g., "audio/wav", "image/png")
     * @return List of compatible features
     */
    public List<SafetensorFeaturePlugin> findFeaturesForInputType(String inputType) {
        return getAvailableFeatures().stream()
                .filter(f -> f.supportedInputTypes().contains(inputType))
                .toList();
    }

    /**
     * Process input through a specific feature.
     * 
     * @param featureId Feature ID
     * @param input     Input data
     * @return Processing result
     */
    public Object process(String featureId, Object input) {
        if (!initialized) {
            throw new IllegalStateException("Feature plugin manager not initialized");
        }

        SafetensorFeaturePlugin feature = features.get(featureId);
        if (feature == null) {
            throw new IllegalArgumentException("Unknown feature: " + featureId);
        }

        if (!isFeatureEnabled(featureId)) {
            throw new IllegalStateException("Feature is disabled: " + featureId);
        }

        if (!feature.isAvailable()) {
            throw new IllegalStateException("Feature is not available: " + featureId);
        }

        LOG.debugf("Processing through feature %s", featureId);
        return feature.process(input);
    }

    /**
     * Process input through the best matching feature.
     * 
     * @param input     Input data
     * @param inputType Input type
     * @return Processing result
     */
    public Object processAuto(Object input, String inputType) {
        List<SafetensorFeaturePlugin> compatibleFeatures = findFeaturesForInputType(inputType);

        if (compatibleFeatures.isEmpty()) {
            throw new IllegalStateException("No compatible feature found for input type: " + inputType);
        }

        // Use highest priority feature
        SafetensorFeaturePlugin feature = compatibleFeatures.get(0);
        LOG.debugf("Auto-selected feature %s for input type %s", feature.id(), inputType);
        return feature.process(input);
    }

    /**
     * Get health status of all features.
     * 
     * @return Map of feature IDs to health status
     */
    public Map<String, Boolean> getHealthStatus() {
        Map<String, Boolean> status = new HashMap<>();
        for (Map.Entry<String, SafetensorFeaturePlugin> entry : features.entrySet()) {
            status.put(entry.getKey(),
                    isFeatureEnabled(entry.getKey()) && entry.getValue().isHealthy());
        }
        return status;
    }

    /**
     * Get feature statistics.
     * 
     * @return Statistics map
     */
    public Map<String, Object> getStats() {
        Map<String, Object> stats = new HashMap<>();
        stats.put("initialized", initialized);
        stats.put("total_features", features.size());
        stats.put("enabled_features", featureStates.values().stream().filter(b -> b).count());
        stats.put("available_features", getAvailableFeatures().size());

        // Add feature details
        List<Map<String, Object>> featureDetails = features.values().stream()
                .map(f -> Map.<String, Object>of(
                        "id", f.id(),
                        "name", f.name(),
                        "version", f.version(),
                        "enabled", isFeatureEnabled(f.id()),
                        "available", f.isAvailable(),
                        "healthy", f.isHealthy(),
                        "models", f.supportedModels()))
                .toList();
        stats.put("features", featureDetails);

        return stats;
    }

    /**
     * Shutdown all features.
     */
    @jakarta.annotation.PreDestroy
    public void shutdown() {
        if (!initialized) {
            return;
        }

        LOG.info("Shutting down feature plugins");

        for (SafetensorFeaturePlugin feature : features.values()) {
            try {
                feature.shutdown();
            } catch (Exception e) {
                LOG.errorf(e, "Error shutting down feature %s", feature.id());
            }
        }

        features.clear();
        featureStates.clear();
        initialized = false;
        LOG.info("Feature plugin manager shutdown complete");
    }
}
