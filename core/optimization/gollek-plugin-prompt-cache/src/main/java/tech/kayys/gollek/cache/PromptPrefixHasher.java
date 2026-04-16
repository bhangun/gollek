package tech.kayys.gollek.cache;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.gollek.kvcache.KVCacheConfig;
import tech.kayys.gollek.cache.PromptCacheConfig;
import tech.kayys.gollek.cache.PrefixHash;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Computes rolling prefix hashes aligned to KV-block boundaries.
 *
 * <p>The key correctness requirement (from the TDS article) is that the shared
 * tokens must start at position 0 of the prompt. This hasher builds a map of
 * {@code prefixLength → hash} at every block-boundary checkpoint, enabling the
 * cache lookup to find the <em>longest matching prefix</em> in O(B) probes where
 * B is the number of block boundaries in the prompt.
 *
 * <h3>Hash algorithms</h3>
 * <ul>
 *   <li>{@code xxhash64}    — fastest, best distribution, recommended for production</li>
 *   <li>{@code murmur3}     — good alternative if xxhash is unavailable</li>
 *   <li>{@code rolling-poly}— simple polynomial rolling hash (no extra dependency)</li>
 * </ul>
 *
 * <p>The hash is incremental: each block boundary hash builds on the previous one,
 * so computing 8 block boundaries costs the same as processing 8 blocks sequentially.
 *
 * <h3>Collision safety</h3>
 * A 64-bit hash over a 16-token block gives a collision probability of ~5×10⁻¹⁹
 * per pair of entries. This is acceptable for a cache where false positives degrade
 * quality slightly (wrong KV blocks injected) rather than cause data loss.
 * A secondary length check ({@code prefixLength}) further reduces risk.
 */
@ApplicationScoped
public class PromptPrefixHasher {

    private static final Logger LOG = Logger.getLogger(PromptPrefixHasher.class);

    // xxhash64 constants (FNV-inspired mixing)
    private static final long XX_PRIME1 = 0x9E3779B185EBCA87L;
    private static final long XX_PRIME2 = 0xC2B2AE3D27D4EB4FL;
    private static final long XX_SEED   = 0x27D4EB2F165667C5L;

    // Polynomial rolling hash constants
    private static final long POLY_BASE = 131L;
    private static final long POLY_MOD  = 0xFFFFFFFFFFFFFFC5L; // large prime

    private final int    blockSize;
    private final String algorithm;

    @Inject
    public PromptPrefixHasher(KVCacheConfig kvConfig, PromptCacheConfig cacheConfig) {
        this.blockSize = kvConfig.getBlockSize();
        this.algorithm = cacheConfig.hashAlgo();
        LOG.infof("[PromptPrefixHasher] init: blockSize=%d, algo=%s", blockSize, algorithm);
    }

    /**
     * Compute a map of {@code prefixLength → hash} for every block boundary
     * in {@code tokenIds}.
     *
     * <p>For example, with {@code blockSize=16} and a 48-token prompt, this returns:
     * <pre>
     *   { 16 → h1, 32 → h2, 48 → h3 }
     * </pre>
     *
     * <p>The caller (lookup plugin) iterates this map from largest to smallest
     * to find the longest cached prefix.
     *
     * @param tokenIds the full token ID array of the incoming prompt
     * @return ordered map of boundary length → 64-bit hash, from smallest to largest
     */
    public Map<Integer, Long> hashBoundaries(int[] tokenIds) {
        if (tokenIds == null || tokenIds.length == 0) return Map.of();

        // LinkedHashMap preserves insertion order (smallest boundary first).
        // We insert in ascending order so callers can iterate in reverse for
        // "longest prefix first" lookups.
        Map<Integer, Long> boundaries = new LinkedHashMap<>();
        long runningHash = initialSeed();

        for (int i = 0; i < tokenIds.length; i++) {
            runningHash = mixToken(runningHash, tokenIds[i]);
            int pos = i + 1;
            if (pos % blockSize == 0) {
                boundaries.put(pos, finalise(runningHash, pos));
            }
        }

        // If the prompt ends mid-block (not on a boundary), we skip the tail —
        // partial blocks are not worth caching because they will be extended
        // immediately on decode and would hash differently across requests.

        return boundaries;
    }

    /**
     * Convenience: return a {@link PrefixHash} for the longest full-block prefix
     * of {@code tokenIds} under the given model and scope.
     *
     * @return empty map entry value if the prompt has no full block
     */
    public Map.Entry<Integer, PrefixHash> longestBoundary(int[] tokenIds, String modelId, String scope) {
        var boundaries = hashBoundaries(tokenIds);
        if (boundaries.isEmpty()) return null;

        // Last entry = longest prefix
        Map.Entry<Integer, Long> last = null;
        for (var e : boundaries.entrySet()) last = e;

        assert last != null;
        return Map.entry(last.getKey(),
                new PrefixHash(modelId, last.getValue(), last.getKey(), scope));
    }

    // -------------------------------------------------------------------------
    // Hash algorithm dispatch
    // -------------------------------------------------------------------------

    private long initialSeed() {
        return switch (algorithm) {
            case "xxhash64", "murmur3" -> XX_SEED;
            default -> POLY_BASE;           // rolling-poly
        };
    }

    private long mixToken(long state, int token) {
        return switch (algorithm) {
            case "xxhash64"    -> xxhash64Mix(state, token);
            case "murmur3"     -> murmur3Mix(state, token);
            default            -> polyMix(state, token);
        };
    }

    private long finalise(long state, int length) {
        return switch (algorithm) {
            case "xxhash64"    -> xxhash64Finalise(state, length);
            case "murmur3"     -> murmur3Finalise(state, length);
            default            -> state ^ (long) length;
        };
    }

    // -------------------------------------------------------------------------
    // xxhash64 — fast, excellent avalanche
    // -------------------------------------------------------------------------

    private static long xxhash64Mix(long acc, int token) {
        long v = Integer.toUnsignedLong(token) * XX_PRIME2;
        acc ^= v;
        acc = Long.rotateLeft(acc, 31) * XX_PRIME1;
        return acc;
    }

    private static long xxhash64Finalise(long acc, int length) {
        acc ^= length;
        acc ^= acc >>> 33;
        acc *= 0xFF51AFD7ED558CCDL;
        acc ^= acc >>> 29;
        acc *= 0xC4CEB9FE1A85EC53L;
        acc ^= acc >>> 32;
        return acc;
    }

    // -------------------------------------------------------------------------
    // Murmur3 (64-bit variant)
    // -------------------------------------------------------------------------

    private static long murmur3Mix(long h, int token) {
        long k = Integer.toUnsignedLong(token);
        k *= 0x87c37b91114253d5L;
        k = Long.rotateLeft(k, 31);
        k *= 0x4cf5ad432745937fL;
        h ^= k;
        h = Long.rotateLeft(h, 27) * 5 + 0x52dce729L;
        return h;
    }

    private static long murmur3Finalise(long h, int length) {
        h ^= length;
        h ^= h >>> 33;
        h *= 0xff51afd7ed558ccdL;
        h ^= h >>> 33;
        h *= 0xc4ceb9fe1a85ec53L;
        h ^= h >>> 33;
        return h;
    }

    // -------------------------------------------------------------------------
    // Simple polynomial rolling hash (zero-dependency fallback)
    // -------------------------------------------------------------------------

    private static long polyMix(long h, int token) {
        return h * POLY_BASE + Integer.toUnsignedLong(token);
    }
}
