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

package tech.kayys.gollek.plugin.runner.gguf.feature;

import java.util.Map;
import java.util.Set;

/**
 * SPI for GGUF feature plugins.
 * 
 * <p>Feature plugins provide domain-specific capabilities for the GGUF runner:
 * <ul>
 *   <li>Text generation (LLMs)</li>
 *   <li>Text embedding</li>
 *   <li>Code generation</li>
 *   <li>Chat/completion</li>
 * </ul>
 * 
 * @since 2.1.0
 */
public interface GGUFFeaturePlugin {

    /**
     * Unique feature plugin identifier.
     */
    String id();

    /**
     * Human-readable name.
     */
    String name();

    /**
     * Feature plugin version.
     */
    default String version() {
        return "1.0.0";
    }

    /**
     * Feature description.
     */
    String description();

    /**
     * Initialize the feature with configuration.
     */
    default void initialize(Map<String, Object> config) {
        // Default: no-op
    }

    /**
     * Check if this feature is available.
     */
    boolean isAvailable();

    /**
     * Get the feature priority. Higher values execute first.
     */
    default int priority() {
        return 0;
    }

    /**
     * Get supported model architectures for this feature.
     */
    Set<String> supportedModels();

    /**
     * Get supported tasks for this feature.
     */
    default Set<String> supportedTasks() {
        return Set.of();
    }

    /**
     * Process input using this feature.
     */
    Object process(Object input);

    /**
     * Get feature-specific metadata.
     */
    default Map<String, Object> metadata() {
        return Map.of();
    }

    /**
     * Shutdown and cleanup resources.
     */
    default void shutdown() {
        // Default: no-op
    }

    /**
     * Check if feature is healthy and operational.
     */
    default boolean isHealthy() {
        return true;
    }
}
