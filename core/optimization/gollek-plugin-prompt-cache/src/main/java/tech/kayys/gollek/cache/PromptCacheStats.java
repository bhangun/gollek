package tech.kayys.gollek.cache;

/**
 * Immutable snapshot of prompt-cache statistics.
 *
 * <p>Returned by {@link PromptCacheStore#stats()} and exposed via
 * Micrometer gauges and the health endpoint.
 */
public record PromptCacheStats(
        long   totalLookups,
        long   hits,
        long   misses,
        long   stores,
        long   evictions,
        long   invalidations,
        long   cachedEntries,
        long   cachedTokens,
        double hitRateLast5Min,   // rolling 5-minute window
        String strategy
) {

    public static PromptCacheStats empty(String strategy) {
        return new PromptCacheStats(0, 0, 0, 0, 0, 0, 0, 0, 0.0, strategy);
    }

    public double hitRate() {
        return totalLookups == 0 ? 0.0 : (double) hits / totalLookups;
    }

    public long savedTokens() {
        // Rough estimate: tokens not re-prefilled = hits * avg prefix length
        // Actual value tracked by the lookup plugin via ExecutionContext metadata.
        return hits;
    }

    @Override
    public String toString() {
        return "PromptCacheStats{strategy=" + strategy
                + ", hitRate=" + String.format("%.1f%%", hitRate() * 100)
                + ", entries=" + cachedEntries
                + ", tokens=" + cachedTokens
                + ", evictions=" + evictions + "}";
    }
}
