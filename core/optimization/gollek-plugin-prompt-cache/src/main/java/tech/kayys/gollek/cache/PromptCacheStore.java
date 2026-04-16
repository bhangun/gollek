package tech.kayys.gollek.cache;

import java.util.Optional;

/**
 * Service Provider Interface for prompt-prefix KV caching.
 *
 * <p>Implementations are selected at runtime via the
 * {@code gollek.cache.prompt.strategy} config property.
 * All implementations must be thread-safe.
 *
 * <p>Lifecycle contract:
 * <ol>
 *   <li>CDI instantiation and {@link #initialize()} called once at startup.</li>
 *   <li>{@link #lookup} / {@link #store} called concurrently from inference threads.</li>
 *   <li>{@link #close()} called on application shutdown.</li>
 * </ol>
 *
 * <p>Implementations must NOT hold strong references to raw KV tensor data.
 * Only block ID metadata is stored here; tensors live in the {@code PhysicalBlockPool}.
 */
public interface PromptCacheStore extends AutoCloseable {

    /**
     * Look up cached KV blocks for the given prefix key.
     *
     * @param hash the prefix key computed by {@code PromptPrefixHasher}
     * @return the cached entry, or empty on miss / eviction / expiry
     */
    Optional<CachedKVEntry> lookup(PrefixHash hash);

    /**
     * Store a newly computed KV prefix so future requests can reuse it.
     *
     * <p>Implementations must not block the calling inference thread for more
     * than a few microseconds. Async writes (e.g. async Redis pipeline) are
     * encouraged for remote backends.
     *
     * @param entry the entry to store; the key is {@code entry.key()}
     */
    void store(CachedKVEntry entry);

    /**
     * Invalidate all entries for a specific model.
     * Called when a model is reloaded, swapped, or unloaded.
     *
     * @param modelId the model identifier
     */
    void invalidateByModel(String modelId);

    /**
     * Invalidate all entries belonging to a session scope.
     * Called on session expiry or user logout.
     *
     * @param sessionId the session identifier (without the {@code session:} prefix)
     */
    void invalidateBySession(String sessionId);

    /**
     * Evict all entries, effectively resetting the cache.
     */
    void invalidateAll();

    /**
     * Return a snapshot of current cache statistics.
     */
    PromptCacheStats stats();

    /**
     * Called once after CDI injection, before any lookups or stores.
     * Implementations should perform lazy pool allocation or connection setup here.
     */
    default void initialize() {}

    /**
     * Return the strategy name for this implementation (e.g. {@code "in-process"}).
     * Used for logging and health-check reporting.
     */
    String strategyName();

    @Override
    default void close() throws Exception {}
}
