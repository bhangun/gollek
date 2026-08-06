package tech.kayys.gollek.cache;

import java.time.Instant;
import java.util.List;
import java.util.Objects;

/**
 * A single prompt-cache entry: the set of physical KV-block IDs that hold
 * the precomputed Key/Value tensors for a token prefix.
 *
 * <p>Block IDs map directly into the {@code PhysicalBlockPool}. The entry
 * is read-only once stored — blocks are pinned via the {@code PagedKVCacheManager}
 * before use and unpinned when the request completes.
 *
 * <p>The {@code hitCount} field is updated atomically by the in-process store
 * to drive LFU eviction. Remote stores (Redis/RocksDB) do not need it.
 */
public record CachedKVEntry(
        PrefixHash    key,
        List<Integer> blockIds,
        int           tokenCount,
        Instant       createdAt,
        Instant       lastAccessedAt,
        long          hitCount,
        String        scope
) {

    public CachedKVEntry {
        Objects.requireNonNull(key,            "key must not be null");
        Objects.requireNonNull(blockIds,       "blockIds must not be null");
        Objects.requireNonNull(createdAt,      "createdAt must not be null");
        Objects.requireNonNull(lastAccessedAt, "lastAccessedAt must not be null");
        Objects.requireNonNull(scope,          "scope must not be null");
        blockIds = List.copyOf(blockIds);
    }

    /** Convenience factory for new entries. */
    public static CachedKVEntry of(PrefixHash key, List<Integer> blockIds, int tokenCount) {
        Instant now = Instant.now();
        return new CachedKVEntry(key, blockIds, tokenCount, now, now, 0L, key.scope());
    }

    /** Returns a copy with incremented hitCount and refreshed lastAccessedAt. */
    public CachedKVEntry accessed() {
        return new CachedKVEntry(key, blockIds, tokenCount, createdAt, Instant.now(), hitCount + 1, scope);
    }

    /** Approximate memory footprint estimate in bytes (metadata only, not KV tensors). */
    public int metadataBytes() {
        return blockIds.size() * 4 + 128; // rough overhead
    }

    @Override
    public String toString() {
        return "CachedKVEntry{model=" + key.modelId()
                + ", tokens=" + tokenCount
                + ", blocks=" + blockIds.size()
                + ", hits=" + hitCount
                + ", scope=" + scope + "}";
    }
}
