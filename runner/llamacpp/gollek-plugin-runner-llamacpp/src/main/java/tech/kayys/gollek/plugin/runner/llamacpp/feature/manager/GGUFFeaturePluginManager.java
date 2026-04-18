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

package tech.kayys.gollek.plugin.runner.llamacpp.feature.manager;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.enterprise.inject.Instance;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.gollek.plugin.runner.llamacpp.feature.GGUFFeaturePlugin;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Manager for GGUF feature plugins.
 * 
 * @since 2.1.0
 */
@ApplicationScoped
public class GGUFFeaturePluginManager {

    private static final Logger LOG = Logger.getLogger(GGUFFeaturePluginManager.class);

    @Inject
    Instance<GGUFFeaturePlugin> featurePluginInstances;

    private final Map<String, GGUFFeaturePlugin> features = new ConcurrentHashMap<>();
    private final Map<String, Boolean> featureStates = new ConcurrentHashMap<>();
    private volatile boolean initialized = false;

    @jakarta.annotation.PostConstruct
    public void initialize() {
        if (initialized) {
            LOG.warn("GGUF feature plugin manager already initialized");
            return;
        }

        LOG.info("Discovering GGUF feature plugins...");

        int[] registered = new int[1];
        int[] failed = new int[1];

        if (featurePluginInstances != null) {
            featurePluginInstances.stream()
                    .forEach(feature -> {
                        try {
                            if (feature.isAvailable()) {
                                register(feature);
                                LOG.infof("✓ Registered GGUF feature plugin: %s", feature.id());
                                registered[0]++;
                            } else {
                                LOG.debugf("Skipping unavailable GGUF feature: %s", feature.id());
                            }
                        } catch (Exception e) {
                            LOG.errorf(e, "Failed to register GGUF feature plugin: %s", feature.id());
                            failed[0]++;
                        }
                    });
        }

        initialized = true;
        LOG.infof("GGUF feature plugin discovery complete. Registered: %d, Failed: %d", registered[0], failed[0]);
    }

    public void register(GGUFFeaturePlugin feature) {
        if (feature == null) {
            throw new IllegalArgumentException("Feature cannot be null");
        }

        LOG.infof("Registering GGUF feature plugin: %s", feature.id());
        features.put(feature.id(), feature);
        featureStates.put(feature.id(), true);
        feature.initialize(Map.of());
    }

    public boolean unregister(String featureId) {
        GGUFFeaturePlugin feature = features.remove(featureId);
        if (feature != null) {
            LOG.infof("Unregistering GGUF feature plugin: %s", featureId);
            feature.shutdown();
            featureStates.remove(featureId);
            return true;
        }
        return false;
    }

    public void setFeatureEnabled(String featureId, boolean enabled) {
        if (features.containsKey(featureId)) {
            featureStates.put(featureId, enabled);
            LOG.infof("GGUF feature %s %s", featureId, enabled ? "enabled" : "disabled");
        }
    }

    public boolean isFeatureEnabled(String featureId) {
        return features.containsKey(featureId) && 
               featureStates.getOrDefault(featureId, true);
    }

    public Optional<GGUFFeaturePlugin> getFeature(String featureId) {
        return Optional.ofNullable(features.get(featureId));
    }

    public List<GGUFFeaturePlugin> getAllFeatures() {
        return new ArrayList<>(features.values());
    }

    public List<GGUFFeaturePlugin> getAvailableFeatures() {
        return features.values().stream()
                .filter(f -> isFeatureEnabled(f.id()))
                .filter(GGUFFeaturePlugin::isAvailable)
                .sorted(Comparator.comparingInt(GGUFFeaturePlugin::priority).reversed())
                .toList();
    }

    public Object process(String featureId, Object input) {
        if (!initialized) {
            throw new IllegalStateException("GGUF feature plugin manager not initialized");
        }

        GGUFFeaturePlugin feature = features.get(featureId);
        if (feature == null) {
            throw new IllegalArgumentException("Unknown GGUF feature: " + featureId);
        }

        if (!isFeatureEnabled(featureId)) {
            throw new IllegalStateException("GGUF feature is disabled: " + featureId);
        }

        if (!feature.isAvailable()) {
            throw new IllegalStateException("GGUF feature is not available: " + featureId);
        }

        LOG.debugf("Processing through GGUF feature %s", featureId);
        return feature.process(input);
    }

    public Map<String, Boolean> getHealthStatus() {
        Map<String, Boolean> status = new HashMap<>();
        for (Map.Entry<String, GGUFFeaturePlugin> entry : features.entrySet()) {
            status.put(entry.getKey(), 
                      isFeatureEnabled(entry.getKey()) && entry.getValue().isHealthy());
        }
        return status;
    }

    public Map<String, Object> getStats() {
        Map<String, Object> stats = new HashMap<>();
        stats.put("initialized", initialized);
        stats.put("total_features", features.size());
        stats.put("enabled_features", featureStates.values().stream().filter(b -> b).count());
        stats.put("available_features", getAvailableFeatures().size());
        
        List<Map<String, Object>> featureDetails = features.values().stream()
                .map(f -> Map.<String, Object>of(
                    "id", f.id(),
                    "name", f.name(),
                    "version", f.version(),
                    "enabled", isFeatureEnabled(f.id()),
                    "available", f.isAvailable(),
                    "healthy", f.isHealthy(),
                    "models", f.supportedModels()
                ))
                .toList();
        stats.put("features", featureDetails);
        
        return stats;
    }

    @jakarta.annotation.PreDestroy
    public void shutdown() {
        if (!initialized) {
            return;
        }

        LOG.info("Shutting down GGUF feature plugins");
        
        for (GGUFFeaturePlugin feature : features.values()) {
            try {
                feature.shutdown();
            } catch (Exception e) {
                LOG.errorf(e, "Error shutting down GGUF feature %s", feature.id());
            }
        }
        
        features.clear();
        featureStates.clear();
        initialized = false;
        LOG.info("GGUF feature plugin manager shutdown complete");
    }
}
