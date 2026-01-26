package tech.kayys.golek.provider.core.mcp;

import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;
import jakarta.enterprise.context.ApplicationScoped;
import org.jboss.logging.Logger;

import java.time.Duration;
import java.util.Optional;

/**
 * LRU cache for MCP resource content.
 * Reduces redundant reads for frequently accessed resources.
 */
@ApplicationScoped
public class MCPResourceCache {

    private static final Logger LOG = Logger.getLogger(MCPResourceCache.class);
    private static final int MAX_CACHE_SIZE = 1000;
    private static final Duration CACHE_TTL = Duration.ofMinutes(15);

    private final Cache<String, MCPResourceContent> cache;

    public MCPResourceCache() {
        this.cache = Caffeine.newBuilder()
                .maximumSize(MAX_CACHE_SIZE)
                .expireAfterWrite(CACHE_TTL)
                .recordStats()
                .build();
    }

    /**
     * Get cached resource
     */
    public Optional<MCPResourceContent> get(String uri) {
        return Optional.ofNullable(cache.getIfPresent(uri));
    }

    /**
     * Put resource in cache
     */
    public void put(String uri, MCPResourceContent content) {
        cache.put(uri, content);
    }

    /**
     * Invalidate specific resource
     */
    public void invalidate(String uri) {
        cache.invalidate(uri);
    }

    /**
     * Clear entire cache
     */
    public void clear() {
        cache.invalidateAll();
        LOG.info("Resource cache cleared");
    }

    /**
     * Get cache statistics
     */
    public CacheStats getStats() {
        var stats = cache.stats();
        return new CacheStats(
                cache.estimatedSize(),
                stats.hitCount(),
                stats.missCount(),
                stats.evictionCount());
    }

    public record CacheStats(
            long size,
            long hits,
            long misses,
            long evictions) {
        public double hitRate() {
            long total = hits + misses;
            return total == 0 ? 0.0 : (double) hits / total;
        }
    }
}