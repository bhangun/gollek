package tech.kayys.gollek.cache;

import io.quarkus.redis.datasource.ReactiveRedisDataSource;
import io.quarkus.redis.datasource.value.ReactiveValueCommands;
import io.smallrye.mutiny.Uni;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import java.time.Duration;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Distributed prompt-cache backed by Redis (or Valkey / Dragonfly).
 *
 * <h3>Design decisions</h3>
 * <ul>
 * <li><b>Async writes</b> — when
 * {@code gollek.cache.prompt.redis.async-writes=true}
 * (default), the store operation is fire-and-forget on the Vert.x event loop.
 * This eliminates Redis write latency from the critical inference path.</li>
 * <li><b>Metadata only</b> — only {@code CachedKVEntry} metadata (block IDs,
 * token count,
 * timestamps, scope) is stored in Redis. The raw KV tensors live in the local
 * {@code PhysicalBlockPool}. On a remote-hit in a multi-node deployment, the
 * receiving node must re-run prefill for its local pool if its blocks differ —
 * this is handled transparently by the lookup plugin's fallback logic.</li>
 * <li><b>Scope invalidation</b> — leverages Redis key patterns:
 * {@code SCAN} + {@code DEL} for model/session invalidation (async).</li>
 * <li><b>TTL delegation</b> — entry TTL is set directly on the Redis key via
 * {@code SETEX} / {@code SET ... EX}. No application-side expiry
 * bookkeeping.</li>
 * </ul>
 *
 * <h3>Key schema</h3>
 * {@code gollek:pc:<scope>:<modelId>:<prefixLength>:<hexHash>}
 */
@ApplicationScoped
@CacheStrategy("redis")
public class RedisPromptCacheStore implements PromptCacheStore {

    private static final Logger LOG = Logger.getLogger(RedisPromptCacheStore.class);

    private final ReactiveRedisDataSource redis;
    private final PromptCacheConfig config;
    private final PromptCacheMetrics metrics;
    private final CacheEntrySerializer serializer;

    private final AtomicLong totalStores = new AtomicLong();
    private final AtomicLong totalEvictions = new AtomicLong();
    private final AtomicLong totalInvalidations = new AtomicLong();

    @Inject
    public RedisPromptCacheStore(
            ReactiveRedisDataSource redis,
            PromptCacheConfig config,
            PromptCacheMetrics metrics,
            CacheEntrySerializer serializer) {
        this.redis = redis;
        this.config = config;
        this.metrics = metrics;
        this.serializer = serializer;
        LOG.infof("[RedisCache] ready: hosts=%s, ttl=%s, asyncWrites=%s",
                config.redis().hosts(), config.ttl(), config.redis().asyncWrites());
    }

    // -------------------------------------------------------------------------
    // PromptCacheStore
    // -------------------------------------------------------------------------

    @Override
    public Optional<CachedKVEntry> lookup(PrefixHash hash) {
        String key = hash.storageKey();
        try {
            String json = valueCommands().get(key)
                    .await().atMost(config.redis().readTimeout());
            if (json == null || json.isBlank()) {
                metrics.recordMiss();
                LOG.debugf("[RedisCache] MISS key=%s", key);
                return Optional.empty();
            }
            CachedKVEntry entry = serializer.deserialize(json);
            metrics.recordHit(entry.tokenCount());
            LOG.debugf("[RedisCache] HIT  key=%s tokens=%d", key, entry.tokenCount());
            return Optional.of(entry.accessed());
        } catch (Exception e) {
            LOG.warnf("[RedisCache] lookup failed for key=%s, treating as miss: %s", key, e.getMessage());
            metrics.recordMiss();
            return Optional.empty();
        }
    }

    @Override
    public void store(CachedKVEntry entry) {
        String key = entry.key().storageKey();
        String value = serializer.serialize(entry);
        Duration ttl = config.ttl();

        Uni<Void> writeOp = ttl.isZero()
                ? valueCommands().set(key, value).replaceWithVoid()
                : valueCommands().setex(key, ttl.getSeconds(), value).replaceWithVoid();

        if (config.redis().asyncWrites()) {
            writeOp.subscribe().with(
                    ignored -> {
                        totalStores.incrementAndGet();
                        metrics.recordStore(entry.tokenCount());
                        LOG.debugf("[RedisCache] STORED (async) key=%s", key);
                    },
                    err -> LOG.warnf("[RedisCache] async store failed key=%s: %s", key, err.getMessage()));
        } else {
            try {
                writeOp.await().atMost(config.redis().readTimeout());
                totalStores.incrementAndGet();
                metrics.recordStore(entry.tokenCount());
                LOG.debugf("[RedisCache] STORED (sync) key=%s", key);
            } catch (Exception e) {
                LOG.warnf("[RedisCache] sync store failed key=%s: %s", key, e.getMessage());
            }
        }
    }

    @Override
    public void invalidateByModel(String modelId) {
        String pattern = config.redis().keyPrefix() + "*:" + modelId + ":*";
        scanAndDelete(pattern, "model=" + modelId);
    }

    @Override
    public void invalidateBySession(String sessionId) {
        String pattern = config.redis().keyPrefix() + "session:" + sessionId + ":*";
        scanAndDelete(pattern, "session=" + sessionId);
    }

    @Override
    public void invalidateAll() {
        String pattern = config.redis().keyPrefix() + "*";
        scanAndDelete(pattern, "ALL");
    }

    @Override
    public PromptCacheStats stats() {
        // Redis does not track hit/miss counts server-side without MONITOR.
        // We return our local counters as an approximation.
        return new PromptCacheStats(
                totalStores.get() + totalInvalidations.get(),
                0L, 0L, // hits/misses tracked only locally in metrics
                totalStores.get(),
                totalEvictions.get(),
                totalInvalidations.get(),
                -1L, // estimatedSize not cheaply available without DBSIZE
                -1L,
                0.0,
                strategyName());
    }

    @Override
    public String strategyName() {
        return "redis";
    }

    // -------------------------------------------------------------------------
    // Internal helpers
    // -------------------------------------------------------------------------

    @SuppressWarnings("unchecked")
    private ReactiveValueCommands<String, String> valueCommands() {
        return redis.value(String.class);
    }

    private void scanAndDelete(String pattern, String label) {
        // Use Quarkus Redis SCAN to avoid blocking O(N) KEYS command
        redis.key(String.class)
                .scan().toMulti()
                .collect().asList()
                .onItem().transformToUni(keys -> {
                    var matching = keys.stream()
                            .filter(k -> k.matches(pattern.replace("*", ".*")))
                            .toList();
                    if (matching.isEmpty())
                        return Uni.createFrom().voidItem();
                    return redis.key(String.class)
                            .del(matching.toArray(String[]::new))
                            .replaceWithVoid();
                })
                .subscribe().with(
                        ignored -> LOG.infof("[RedisCache] invalidated %s", label),
                        err -> LOG.warnf("[RedisCache] invalidation failed %s: %s", label, err.getMessage()));
        totalInvalidations.incrementAndGet();
    }
}
