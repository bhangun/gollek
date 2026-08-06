package tech.kayys.gollek.cache;

import java.util.Objects;

/**
 * Immutable key that uniquely identifies a cached prompt prefix.
 *
 * <p>The hash is computed over the token ID sequence up to {@code prefixLength},
 * aligned to KV block boundaries. Two requests share a cache entry if and only
 * if they have the same model and identical token IDs from position 0 through
 * {@code prefixLength}, where that length falls on a block boundary.
 *
 * <p>Scope partitions the cache namespace:
 * <ul>
 *   <li>{@code global}       — shared across all users and sessions</li>
 *   <li>{@code model:<id>}   — scoped to a single model identity</li>
 *   <li>{@code session:<id>} — scoped to a single user session</li>
 * </ul>
 */
public record PrefixHash(
        String modelId,
        long   tokenHash,
        int    prefixLength,
        String scope
) {

    public PrefixHash {
        Objects.requireNonNull(modelId, "modelId must not be null");
        Objects.requireNonNull(scope,   "scope must not be null");
        if (prefixLength <= 0) throw new IllegalArgumentException("prefixLength must be > 0");
    }

    /** Convenience constructor defaulting to {@code global} scope. */
    public PrefixHash(String modelId, long tokenHash, int prefixLength) {
        this(modelId, tokenHash, prefixLength, "global");
    }

    /**
     * Storage key for Redis/RocksDB/disk backends.
     * Format: {@code gollek:pc:<scope>:<modelId>:<prefixLength>:<hexHash>}
     */
    public String storageKey() {
        return "gollek:pc:" + scope + ":" + modelId + ":"
                + prefixLength + ":" + Long.toUnsignedString(tokenHash, 16);
    }

    public PrefixHash withScope(String newScope) {
        return new PrefixHash(modelId, tokenHash, prefixLength, newScope);
    }
}
