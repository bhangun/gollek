/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * QuantizerRegistry.java
 * ──────────────────────
 * Central registry for all quantizer implementations.
 * Provides unified access to GPTQ, AWQ, AutoRound, and TurboQuant.
 */
package tech.kayys.gollek.safetensor.quantization;

import tech.kayys.gollek.safetensor.quantization.quantizer.GPTQQuantizerAdapter;
import tech.kayys.gollek.safetensor.quantization.quantizer.Quantizer;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Registry of all available quantizer implementations.
 * <p>
 * Provides a central access point for quantizers, supporting:
 * <ul>
 * <li>Discovery of available quantization methods</li>
 * <li>Automatic quantizer selection based on config</li>
 * <li>Registration of custom quantizers</li>
 * </ul>
 *
 * <h3>Supported Quantizers</h3>
 * <ul>
 * <li><b>GPTQ</b> - Hessian-based per-layer quantization</li>
 * <li><b>AWQ</b> - Activation-aware weight quantization</li>
 * <li><b>AutoRound</b> - Optimization-based rounding + scaling</li>
 * <li><b>TurboQuant</b> - Online vector quantization with random rotation</li>
 * </ul>
 *
 * <h3>Example</h3>
 * <pre>{@code
 * // Get all available quantizers
 * Map<String, Quantizer> quantizers = QuantizerRegistry.getAll();
 *
 * // Get specific quantizer by name
 * Quantizer gptq = QuantizerRegistry.get("GPTQ");
 *
 * // Get best quantizer for config
 * Quantizer best = QuantizerRegistry.selectBest(config);
 * }</pre>
 */
public final class QuantizerRegistry {

    private static final Map<String, Quantizer> QUANTIZERS = new ConcurrentHashMap<>();

    static {
        // Register built-in quantizers
        register("GPTQ", new GPTQQuantizerAdapter());
        // TODO: Add AWQ, AutoRound, TurboQuant adapters when tensor-level APIs are ready
    }

    private QuantizerRegistry() {
        // Utility class
    }

    /**
     * Register a quantizer implementation.
     *
     * @param name      unique quantizer name
     * @param quantizer quantizer instance
     */
    public static void register(String name, Quantizer quantizer) {
        Objects.requireNonNull(name, "Quantizer name must not be null");
        Objects.requireNonNull(quantizer, "Quantizer instance must not be null");
        QUANTIZERS.put(name.toLowerCase(), quantizer);
    }

    /**
     * Get a quantizer by name.
     *
     * @param name quantizer name (case-insensitive)
     * @return quantizer instance, or null if not found
     */
    public static Quantizer get(String name) {
        return QUANTIZERS.get(name.toLowerCase());
    }

    /**
     * Get all registered quantizers.
     *
     * @return unmodifiable map of name → quantizer
     */
    public static Map<String, Quantizer> getAll() {
        return Collections.unmodifiableMap(QUANTIZERS);
    }

    /**
     * Get all registered quantizer names.
     *
     * @return set of quantizer names
     */
    public static Set<String> getNames() {
        return Collections.unmodifiableSet(QUANTIZERS.keySet());
    }

    /**
     * Check if a quantizer is registered.
     *
     * @param name quantizer name
     * @return true if registered
     */
    public static boolean has(String name) {
        return QUANTIZERS.containsKey(name.toLowerCase());
    }

    /**
     * Select the best quantizer for the given configuration.
     * <p>
     * Selection criteria:
     * <ol>
     * <li>Quantizers that explicitly support the config</li>
     * <li>Prefer quantizers with higher compression ratios</li>
     * <li>Fall back to first available quantizer</li>
     * </ol>
     *
     * @param config quantization configuration
     * @return best quantizer, or null if none available
     */
    public static Quantizer selectBest(QuantConfig config) {
        List<Quantizer> candidates = new ArrayList<>();

        for (Quantizer q : QUANTIZERS.values()) {
            if (q.supports(config)) {
                candidates.add(q);
            }
        }

        if (candidates.isEmpty()) {
            return null;
        }

        // Return first matching quantizer for now
        // TODO: Implement smarter selection based on compression ratio, speed, etc.
        return candidates.get(0);
    }

    /**
     * Clear all registered quantizers.
     * <p>
     * Useful for testing or reinitialization.
     */
    public static void clear() {
        QUANTIZERS.clear();
    }

    /**
     * Get quantizer statistics.
     *
     * @return map of quantizer name to metadata
     */
    public static Map<String, String> getStats() {
        Map<String, String> stats = new LinkedHashMap<>();
        for (var entry : QUANTIZERS.entrySet()) {
            Quantizer q = entry.getValue();
            stats.put(entry.getKey(), String.format("%s (supports=%s)",
                    q.getName(), q.supports(QuantConfig.int4Gptq())));
        }
        return stats;
    }
}
