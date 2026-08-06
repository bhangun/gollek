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

import java.util.Map;
import java.util.Set;

/**
 * SPI for GPU kernel optimization plugins.
 * 
 * <p>Optimization plugins enhance inference performance by providing:</p>
 * <ul>
 *   <li>Custom attention kernels (FlashAttention, PagedAttention, etc.)</li>
 *   <li>KV cache optimizations</li>
 *   <li>Memory management improvements</li>
 *   <li>Execution scheduling optimizations</li>
 * </ul>
 * 
 * @since 2.1.0
 */
public interface OptimizationPlugin {

    /**
     * Unique plugin identifier.
     */
    String id();

    /**
     * Human-readable name.
     */
    String name();

    /**
     * Plugin version.
     */
    default String version() {
        return "1.0.0";
    }

    /**
     * Plugin description.
     */
    String description();

    /**
     * Initialize the plugin with configuration.
     */
    default void initialize(Map<String, Object> config) {
        // Default: no-op
    }

    /**
     * Check if this optimization is available on the current hardware.
     */
    boolean isAvailable();

    /**
     * Get the optimization priority. Higher values execute first.
     */
    default int priority() {
        return 0;
    }

    /**
     * Get supported model architectures.
     */
    default Set<String> supportedArchitectures() {
        return Set.of();
    }

    /**
     * Get supported GPU architectures.
     */
    default Set<String> supportedGpuArchs() {
        return Set.of();
    }

    /**
     * Apply the optimization.
     */
    boolean apply(ExecutionContext context);

    /**
     * Get optimization-specific metadata.
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
     * Check if plugin is healthy and operational.
     */
    default boolean isHealthy() {
        return true;
    }
}
