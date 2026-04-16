package tech.kayys.gollek.cache;

import io.smallrye.config.ConfigMapping;
import io.smallrye.config.WithDefault;
import io.smallrye.config.WithName;

import java.time.Duration;
import java.util.Optional;

/**
 * Config-driven control of all prompt-cache behaviour.
 *
 * <p>All properties are under the {@code gollek.cache.prompt} prefix.
 * Example {@code application.properties}:
 * <pre>
 * # --- Core ---
 * gollek.cache.prompt.enabled=true
 * gollek.cache.prompt.strategy=in-process      # in-process | redis | disk | noop
 * gollek.cache.prompt.scope=global             # global | model | session
 * gollek.cache.prompt.min-cacheable-tokens=64  # skip tiny prompts
 *
 * # --- Eviction ---
 * gollek.cache.prompt.max-entries=4096
 * gollek.cache.prompt.max-total-tokens=524288  # ~512K tokens across all entries
 * gollek.cache.prompt.ttl=PT30M               # ISO-8601; 0 = no expiry
 * gollek.cache.prompt.eviction-policy=lru      # lru | lfu | ttl-only
 *
 * # --- In-process ---
 * gollek.cache.prompt.in-process.initial-capacity=256
 * gollek.cache.prompt.in-process.record-stats=true
 *
 * # --- Redis ---
 * gollek.cache.prompt.redis.hosts=localhost:6379
 * gollek.cache.prompt.redis.key-prefix=gollek:pc:
 * gollek.cache.prompt.redis.max-total-connections=16
 * gollek.cache.prompt.redis.read-timeout=PT1S
 * gollek.cache.prompt.redis.async-writes=true
 *
 * # --- Disk / RocksDB ---
 * gollek.cache.prompt.disk.path=/var/gollek/prompt-cache
 * gollek.cache.prompt.disk.backend=mmap       # mmap | rocksdb
 * gollek.cache.prompt.disk.max-size-mb=4096
 * gollek.cache.prompt.disk.sync-on-write=false
 *
 * # --- Advanced ---
 * gollek.cache.prompt.hash-algo=xxhash64      # xxhash64 | murmur3 | rolling-poly
 * gollek.cache.prompt.warm-on-startup=false
 * gollek.cache.prompt.warm-model-ids=         # comma-separated
 * gollek.cache.prompt.async-store=true        # store computed blocks async
 * </pre>
 */
@ConfigMapping(prefix = "gollek.cache.prompt")
public interface PromptCacheConfig {

    // -------------------------------------------------------------------------
    // Core
    // -------------------------------------------------------------------------

    @WithDefault("true")
    boolean enabled();

    /** Storage backend: {@code in-process}, {@code redis}, {@code disk}, {@code noop}. */
    @WithDefault("in-process")
    String strategy();

    /**
     * Cache namespace scope.
     * {@code global} = all users share one cache.
     * {@code model}  = per loaded model.
     * {@code session}= per user session (chat history).
     */
    @WithDefault("global")
    String scope();

    /** Minimum prefix token count to consider caching. Avoids polluting cache with tiny prompts. */
    @WithName("min-cacheable-tokens")
    @WithDefault("64")
    int minCacheableTokens();

    // -------------------------------------------------------------------------
    // Eviction
    // -------------------------------------------------------------------------

    @WithName("max-entries")
    @WithDefault("4096")
    int maxEntries();

    /**
     * Hard cap on total cached tokens across all entries.
     * Default 512K. When exceeded, LRU/LFU eviction kicks in regardless of TTL.
     */
    @WithName("max-total-tokens")
    @WithDefault("524288")
    long maxTotalTokens();

    /**
     * Time-to-live per entry. {@code PT0S} means no expiry (rely on size-based eviction only).
     */
    @WithDefault("PT30M")
    Duration ttl();

    /**
     * Eviction policy: {@code lru}, {@code lfu}, or {@code ttl-only}.
     * {@code lru} and {@code lfu} also respect TTL as a secondary eviction trigger.
     */
    @WithName("eviction-policy")
    @WithDefault("lru")
    String evictionPolicy();

    // -------------------------------------------------------------------------
    // In-Process sub-config
    // -------------------------------------------------------------------------

    @WithName("in-process")
    InProcess inProcess();

    interface InProcess {
        @WithName("initial-capacity")
        @WithDefault("256")
        int initialCapacity();

        @WithName("record-stats")
        @WithDefault("true")
        boolean recordStats();
    }

    // -------------------------------------------------------------------------
    // Redis sub-config
    // -------------------------------------------------------------------------

    Redis redis();

    interface Redis {
        /** Comma-separated list of host:port pairs. Supports Sentinel and Cluster URLs. */
        @WithDefault("localhost:6379")
        String hosts();

        @WithName("key-prefix")
        @WithDefault("gollek:pc:")
        String keyPrefix();

        @WithName("max-total-connections")
        @WithDefault("16")
        int maxTotalConnections();

        @WithName("read-timeout")
        @WithDefault("PT1S")
        Duration readTimeout();

        /** Write KV metadata to Redis asynchronously to avoid adding latency on cache-miss path. */
        @WithName("async-writes")
        @WithDefault("true")
        boolean asyncWrites();

        /** Serialize block metadata as JSON or CBOR. */
        @WithDefault("json")
        String serializer();
    }

    // -------------------------------------------------------------------------
    // Disk sub-config
    // -------------------------------------------------------------------------

    Disk disk();

    interface Disk {
        @WithDefault("/var/gollek/prompt-cache")
        String path();

        /** {@code mmap} for simple memory-mapped files; {@code rocksdb} for persistent KV store. */
        @WithDefault("mmap")
        String backend();

        @WithName("max-size-mb")
        @WithDefault("4096")
        long maxSizeMb();

        @WithName("sync-on-write")
        @WithDefault("false")
        boolean syncOnWrite();
    }

    // -------------------------------------------------------------------------
    // Advanced / Hashing
    // -------------------------------------------------------------------------

    /** Hash algorithm: {@code xxhash64} (default, fastest), {@code murmur3}, {@code rolling-poly}. */
    @WithName("hash-algo")
    @WithDefault("xxhash64")
    String hashAlgo();

    /**
     * If true, pre-warm the cache for configured models at startup
     * using any persisted disk snapshot.
     */
    @WithName("warm-on-startup")
    @WithDefault("false")
    boolean warmOnStartup();

    /** Model IDs to warm on startup, comma-separated. Empty = warm all. */
    @WithName("warm-model-ids")
    @WithDefault("")
    Optional<String> warmModelIds();

    /**
     * If true, persisting new entries to the store is done on a background thread,
     * keeping the inference hot path free. Safe because the KV data is already
     * in the PhysicalBlockPool; metadata write can be slightly delayed.
     */
    @WithName("async-store")
    @WithDefault("true")
    boolean asyncStore();
}
