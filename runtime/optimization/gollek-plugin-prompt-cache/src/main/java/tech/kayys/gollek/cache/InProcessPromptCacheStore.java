package tech.kayys.gollek.cache;

import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;
import com.github.benmanes.caffeine.cache.RemovalCause;
import com.github.benmanes.caffeine.cache.stats.CacheStats;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.gollek.cache.PromptCacheConfig;
import tech.kayys.gollek.cache.PromptCacheMetrics;
import tech.kayys.gollek.cache.CacheStrategy;
import tech.kayys.gollek.cache.CachedKVEntry;
import tech.kayys.gollek.cache.PrefixHash;
import tech.kayys.gollek.cache.PromptCacheStats;
import tech.kayys.gollek.cache.PromptCacheStore;

import java.util.Optional;
import java.util.concurrent.atomic.AtomicLong;

/**
 * In-process prompt-cache backed by Caffeine.
 *
 * <p>Suitable for single-JVM standalone deployments where low latency
 * is the primary concern. No network round-trips; lookup cost is O(1)
 * hash table access (~100–300 ns typical).
 *
 * <h3>Eviction</h3>
 * <ul>
 *   <li><b>LRU</b> — {@code maximumSize} cap with access-order eviction.</li>
 *   <li><b>LFU</b> — {@code maximumSize} + {@code weigher} on token count,
 *       Caffeine's TinyLFU approximation.</li>
 *   <li><b>TTL-only</b> — no size cap, expiry drives eviction.</li>
 * </ul>
 *
 * <p>Scope invalidation (by model or session) is O(n) because Caffeine does not
 * support partial iteration. For large caches (&gt;50K entries) the Redis or
 * disk backends are recommended when frequent model swaps are expected.
 */
@ApplicationScoped
@CacheStrategy("in-process")
public class InProcessPromptCacheStore implements PromptCacheStore {

    private static final Logger LOG = Logger.getLogger(InProcessPromptCacheStore.class);

    private final Cache<String, CachedKVEntry> cache;
    private final PromptCacheConfig            config;
    private final PromptCacheMetrics           metrics;

    // Running totals
    private final AtomicLong totalStores       = new AtomicLong();
    private final AtomicLong totalEvictions     = new AtomicLong();
    private final AtomicLong totalInvalidations = new AtomicLong();
    private final AtomicLong totalTokens        = new AtomicLong();

    @Inject
    public InProcessPromptCacheStore(PromptCacheConfig config, PromptCacheMetrics metrics) {
        this.config  = config;
        this.metrics = metrics;
        this.cache   = buildCache(config);
        LOG.infof("[InProcessCache] ready: policy=%s, maxEntries=%d, ttl=%s",
                config.evictionPolicy(), config.maxEntries(), config.ttl());
    }

    // -------------------------------------------------------------------------
    // PromptCacheStore
    // -------------------------------------------------------------------------

    @Override
    public Optional<CachedKVEntry> lookup(PrefixHash hash) {
        CachedKVEntry entry = cache.getIfPresent(hash.storageKey());
        if (entry != null) {
            CachedKVEntry updated = entry.accessed();
            cache.put(hash.storageKey(), updated);
            metrics.recordHit(entry.tokenCount());
            LOG.debugf("[InProcessCache] HIT  key=%s tokens=%d hits=%d",
                    hash.storageKey(), entry.tokenCount(), updated.hitCount());
            return Optional.of(updated);
        }
        metrics.recordMiss();
        LOG.debugf("[InProcessCache] MISS key=%s", hash.storageKey());
        return Optional.empty();
    }

    @Override
    public void store(CachedKVEntry entry) {
        cache.put(entry.key().storageKey(), entry);
        totalStores.incrementAndGet();
        totalTokens.addAndGet(entry.tokenCount());
        metrics.recordStore(entry.tokenCount());
        LOG.debugf("[InProcessCache] STORE key=%s tokens=%d", entry.key().storageKey(), entry.tokenCount());
    }

    @Override
    public void invalidateByModel(String modelId) {
        long before = cache.estimatedSize();
        cache.asMap().keySet().removeIf(k -> k.contains(":" + modelId + ":"));
        long removed = before - cache.estimatedSize();
        totalInvalidations.addAndGet(removed);
        LOG.infof("[InProcessCache] invalidated ~%d entries for model=%s", removed, modelId);
    }

    @Override
    public void invalidateBySession(String sessionId) {
        long before = cache.estimatedSize();
        cache.asMap().keySet().removeIf(k -> k.contains("session:" + sessionId));
        long removed = before - cache.estimatedSize();
        totalInvalidations.addAndGet(removed);
        LOG.infof("[InProcessCache] invalidated ~%d entries for session=%s", removed, sessionId);
    }

    @Override
    public void invalidateAll() {
        long count = cache.estimatedSize();
        cache.invalidateAll();
        totalInvalidations.addAndGet(count);
        totalTokens.set(0);
        LOG.infof("[InProcessCache] invalidated all (~%d entries)", count);
    }

    @Override
    public PromptCacheStats stats() {
        CacheStats cs = config.inProcess().recordStats() ? cache.stats() : CacheStats.empty();
        return new PromptCacheStats(
                cs.requestCount(),
                cs.hitCount(),
                cs.missCount(),
                totalStores.get(),
                totalEvictions.get(),
                totalInvalidations.get(),
                cache.estimatedSize(),
                totalTokens.get(),
                cs.hitRate(),
                strategyName()
        );
    }

    @Override
    public String strategyName() { return "in-process"; }

    @Override
    public void close() {
        cache.invalidateAll();
        cache.cleanUp();
    }

    // -------------------------------------------------------------------------
    // Cache construction
    // -------------------------------------------------------------------------

    private Cache<String, CachedKVEntry> buildCache(PromptCacheConfig cfg) {
        Caffeine<Object, Object> builder = Caffeine.newBuilder();

        // TTL
        if (!cfg.ttl().isZero()) {
            builder.expireAfterWrite(cfg.ttl());
        }

        // Eviction policy
        switch (cfg.evictionPolicy()) {
            case "lfu" -> builder
                    .maximumWeight(cfg.maxTotalTokens())
                    .weigher((String k, CachedKVEntry v) -> Math.max(1, v.tokenCount()));
            case "ttl-only" -> {
                // no size cap — TTL is the only eviction mechanism
            }
            default -> builder.maximumSize(cfg.maxEntries()); // lru (default)
        }

        // Stats
        if (cfg.inProcess().recordStats()) {
            builder.recordStats();
        }

        // Eviction listener — fires on size eviction, TTL, or explicit invalidation
        builder.removalListener((String key, CachedKVEntry entry, RemovalCause cause) -> {
            if (entry != null) {
                totalEvictions.incrementAndGet();
                totalTokens.addAndGet(-entry.tokenCount());
                metrics.recordEviction();
                LOG.debugf("[InProcessCache] EVICT key=%s cause=%s tokens=%d",
                        key, cause, entry.tokenCount());
            }
        });

        return builder
                .initialCapacity(cfg.inProcess().initialCapacity())
                .build();
    }
}
