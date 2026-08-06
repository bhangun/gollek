/*
 * Copyright (c) 2026 Kayys.tech — Gollek Runtime SPI
 */
package tech.kayys.gollek.spi.runtime;

/**
 * Cost and performance profile for an {@link ExecutionProvider}.
 *
 * <p>The intelligent router uses these values in the scoring formula:</p>
 * <pre>
 * score = latencyWeight × normalized(latencyMs)
 *       + costWeight    × normalized(costPer1KTokens)
 *       + qualityWeight × normalized(1 − qualityScore)
 * </pre>
 *
 * @param costPer1KTokens       estimated USD cost per 1 000 tokens
 * @param latencyMs             average end-to-end latency in milliseconds
 * @param throughputTokensPerSec sustained throughput (tokens / second)
 * @param qualityScore          quality heuristic in [0.0, 1.0] (1.0 = best)
 */
public record CostProfile(
        double costPer1KTokens,
        double latencyMs,
        double throughputTokensPerSec,
        double qualityScore) {

    /**
     * A zero-cost profile for local inference with unknown performance.
     */
    public static CostProfile local() {
        return new CostProfile(0.0, 0.0, 0.0, 1.0);
    }

    /**
     * Create a profile for a remote API provider.
     */
    public static CostProfile remote(double costPer1K, double latencyMs) {
        return new CostProfile(costPer1K, latencyMs, 0.0, 0.9);
    }
}
