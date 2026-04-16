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

import java.util.Collections;
import java.util.Map;
import java.util.Set;

/**
 * SPI for optimization plugins.
 * 
 * <p>Optimization plugins provide performance enhancements for runner plugins:
 * <ul>
 *   <li>FlashAttention-3/4 - Optimized attention kernels</li>
 *   <li>PagedAttention - Efficient KV cache management</li>
 *   <li>KV Cache - KV cache optimizations</li>
 *   <li>Prompt Cache - Prompt caching for repeated queries</li>
 * </ul>
 * 
 * <h2>Integration with Runners</h2>
 * <pre>{@code
 * Runner Plugin (GGUF, Safetensor, etc.)
 *     ↓
 * Feature Plugin (Audio, Vision, Text)
 *     ↓
 * Optimization Plugin (FA3, PagedAttn, etc.)
 *     ↓
 * Native Backend (llama.cpp, etc.)
 * }</pre>
 * 
 * @since 2.1.0
 */
public interface OptimizationPlugin {

    /**
     * Unique optimization plugin identifier.
     * 
     * @return optimization ID (e.g., "flash-attention-3", "paged-attention")
     */
    String id();

    /**
     * Human-readable name.
     * 
     * @return optimization name
     */
    String name();

    /**
     * Optimization version.
     * 
     * @return semantic version
     */
    default String version() {
        return "1.0.0";
    }

    /**
     * Optimization description.
     * 
     * @return description
     */
    String description();

    /**
     * Initialize the optimization with configuration.
     * 
     * @param config Configuration parameters
     */
    default void initialize(Map<String, Object> config) {
        // Default: no-op
    }

    /**
     * Check if this optimization is available on current hardware.
     * 
     * @return true if optimization can be applied
     */
    boolean isAvailable();

    /**
     * Get the optimization priority. Higher values execute first.
     * 
     * @return priority value (default: 0)
     */
    default int priority() {
        return 0;
    }

    /**
     * Get compatible runner IDs.
     * 
     * @return set of compatible runner IDs (e.g., "gguf-runner", "safetensor-runner")
     */
    Set<String> compatibleRunners();

    /**
     * Get supported GPU architectures.
     * 
     * @return set of supported GPU architectures (e.g., "hopper", "blackwell", "ampere")
     */
    Set<String> supportedGpuArchs();

    /**
     * Apply the optimization.
     * 
     * @param context Optimization context with model parameters and buffers
     * @return true if optimization was successfully applied
     */
    boolean apply(OptimizationContext context);

    /**
     * Get optimization-specific metadata.
     * 
     * @return metadata map
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
     * Check if optimization is healthy and operational.
     * 
     * @return true if optimization is healthy
     */
    default boolean isHealthy() {
        return true;
    }

    /**
     * Validate optimization prerequisites before initialization.
     *
     * <p>Called during plugin validation phase to ensure all requirements
     * are met (hardware, libraries, compatibility).</p>
     *
     * @return validation result with status and any error messages
     */
    default OptimizationValidationResult validate() {
        try {
            boolean available = isAvailable();
            if (available) {
                return OptimizationValidationResult.valid();
            } else {
                return OptimizationValidationResult.invalid(
                    "Optimization not available on this platform");
            }
        } catch (Exception e) {
            return OptimizationValidationResult.invalid(
                "Validation failed: " + e.getMessage());
        }
    }

    /**
     * Initialize the optimization with context and configuration.
     *
     * <p>Called during plugin initialization phase. Should prepare all
     * resources needed for optimization application.</p>
     *
     * @param context Optimization context with configuration
     * @throws OptimizationException if initialization fails
     */
    default void initialize(OptimizationContext context) throws OptimizationException {
        // Default: no-op
    }

    /**
     * Check if optimization is healthy and operational.
     *
     * <p>Called periodically during runtime to monitor optimization health.</p>
     *
     * @return health status with details
     */
    default OptimizationHealth health() {
        boolean healthy = isHealthy();
        return healthy ?
            OptimizationHealth.healthy() :
            OptimizationHealth.unhealthy("Optimization reported unhealthy");
    }

    /**
     * Get supported operations for this optimization.
     *
     * @return set of supported operation names
     */
    default Set<String> supportedOperations() {
        return Collections.emptySet();
    }

    /**
     * Get optimization configuration.
     *
     * @return optimization configuration
     */
    default OptimizationConfig getConfig() {
        return OptimizationConfig.defaultConfig();
    }

    /**
     * Check if a class is available in the classpath.
     *
     * @param className fully qualified class name
     * @return true if class can be loaded
     */
    static boolean isClassAvailable(String className) {
        try {
            Class.forName(className);
            return true;
        } catch (ClassNotFoundException | NoClassDefFoundError e) {
            return false;
        }
    }

    /**
     * Optimization context provided to plugins.
     */
    interface OptimizationContext {
        /**
         * Get model parameter.
         */
        <T> T getParameter(String key, Class<T> type);

        /**
         * Get model parameter with default.
         */
        default <T> T getParameter(String key, Class<T> type, T defaultValue) {
            T value = getParameter(key, type);
            return value != null ? value : defaultValue;
        }

        /**
         * Get GPU device ID.
         */
        int getDeviceId();

        /**
         * Check if running on GPU.
         */
        boolean isGpu();

        /**
         * Get GPU architecture.
         */
        String getGpuArch();

        /**
         * Get compute capability (NVIDIA).
         */
        default int getComputeCapability() {
            return 0;
        }

        /**
         * Get optimization configuration.
         */
        OptimizationConfig getConfig();

        /**
         * Get operation ID.
         */
        default String getOperationId() {
            return "default-operation";
        }
    }
}
