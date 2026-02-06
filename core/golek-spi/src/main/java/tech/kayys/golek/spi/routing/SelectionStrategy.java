package tech.kayys.golek.spi.routing;

/**
 * Provider selection strategies for multi-provider routing.
 * Determines how the router selects among available providers.
 */
public enum SelectionStrategy {

    /**
     * Cycle through providers sequentially.
     * Each request goes to the next provider in order.
     */
    ROUND_ROBIN,

    /**
     * Random selection with equal probability.
     */
    RANDOM,

    /**
     * Random selection with configurable weights per provider.
     * Higher weight = higher selection probability.
     */
    WEIGHTED_RANDOM,

    /**
     * Prefer providers with lowest current load/queue depth.
     */
    LEAST_LOADED,

    /**
     * Prefer cheapest providers (local over cloud).
     * Optimizes for cost when quality is acceptable.
     */
    COST_OPTIMIZED,

    /**
     * Prefer fastest providers based on P95 latency metrics.
     */
    LATENCY_OPTIMIZED,

    /**
     * Use explicit user-specified provider.
     * Fails if preferred provider is unavailable.
     */
    USER_SELECTED,

    /**
     * Primary provider with automatic failover.
     * Tries primary first, switches on quota/error.
     */
    FAILOVER,

    /**
     * Score-based selection using multi-factor algorithm.
     * Considers performance, cost, latency, reliability.
     */
    SCORED;

    /**
     * Check if strategy requires provider weights configuration.
     */
    public boolean requiresWeights() {
        return this == WEIGHTED_RANDOM;
    }

    /**
     * Check if strategy can automatically failover on errors.
     */
    public boolean supportsFailover() {
        return this == FAILOVER || this == SCORED;
    }

    /**
     * Check if strategy respects user preferences.
     */
    public boolean respectsUserPreference() {
        return this == USER_SELECTED || this == SCORED;
    }
}
