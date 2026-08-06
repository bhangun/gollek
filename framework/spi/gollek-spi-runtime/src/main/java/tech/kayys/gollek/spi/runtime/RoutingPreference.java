/*
 * Copyright (c) 2026 Kayys.tech — Gollek Runtime SPI
 */
package tech.kayys.gollek.spi.runtime;

/**
 * Weights that control the intelligent router's scoring formula.
 *
 * <p>Example profiles:</p>
 * <ul>
 *   <li><b>Low-latency (real-time chat):</b>
 *       {@code new RoutingPreference(0.7, 0.1, 0.2, true)}</li>
 *   <li><b>Cost-optimised (batch processing):</b>
 *       {@code new RoutingPreference(0.1, 0.7, 0.2, false)}</li>
 *   <li><b>Quality-first (critical outputs):</b>
 *       {@code new RoutingPreference(0.1, 0.1, 0.8, false)}</li>
 * </ul>
 *
 * @param latencyWeight weight for latency in [0.0, 1.0]
 * @param costWeight    weight for cost in [0.0, 1.0]
 * @param qualityWeight weight for quality in [0.0, 1.0]
 * @param preferLocal   if true, local providers receive a scoring bonus
 */
public record RoutingPreference(
        double latencyWeight,
        double costWeight,
        double qualityWeight,
        boolean preferLocal) {

    /**
     * Balanced default: equal latency/cost/quality weights, prefer local.
     */
    public static final RoutingPreference BALANCED =
            new RoutingPreference(0.34, 0.33, 0.33, true);

    /**
     * Optimise for lowest latency (real-time use cases).
     */
    public static final RoutingPreference LOW_LATENCY =
            new RoutingPreference(0.7, 0.1, 0.2, true);

    /**
     * Optimise for lowest cost (batch / background use cases).
     */
    public static final RoutingPreference LOW_COST =
            new RoutingPreference(0.1, 0.7, 0.2, false);

    /**
     * Optimise for highest quality output.
     */
    public static final RoutingPreference HIGH_QUALITY =
            new RoutingPreference(0.1, 0.1, 0.8, false);
}
