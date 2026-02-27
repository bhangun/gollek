package tech.kayys.gollek.inference.libtorch;

import org.jboss.logging.Logger;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * KV-Cache manager for TorchScript inference.
 * <p>
 * Maintains key-value caches across sequential token generation steps
 * to avoid redundant recomputation of attention scores for prior tokens.
 * Each session gets its own cache, keyed by session ID.
 * <p>
 * Thread-safe: each cache is session-local; concurrent sessions
 * use separate cache entries.
 */
public class KVCacheManager {

    private static final Logger log = Logger.getLogger(KVCacheManager.class);

    private final Map<String, KVCache> caches = new ConcurrentHashMap<>();
    private final int maxCacheSize;

    public KVCacheManager(int maxCacheSize) {
        this.maxCacheSize = maxCacheSize;
    }

    /**
     * Get or create a KV-Cache for the given session.
     *
     * @param sessionId unique session identifier
     * @param numLayers number of transformer layers
     * @param numHeads  number of attention heads per layer
     * @param headDim   dimension per attention head
     * @return the cache for this session
     */
    public KVCache getOrCreate(String sessionId, int numLayers, int numHeads, int headDim) {
        return caches.computeIfAbsent(sessionId,
                id -> {
                    if (caches.size() >= maxCacheSize) {
                        evictOldest();
                    }
                    log.debugf("Creating KV-Cache for session=%s (layers=%d, heads=%d, dim=%d)",
                            sessionId, numLayers, numHeads, headDim);
                    return new KVCache(numLayers, numHeads, headDim);
                });
    }

    /**
     * Invalidate and release the cache for a session.
     */
    public void invalidate(String sessionId) {
        KVCache removed = caches.remove(sessionId);
        if (removed != null) {
            removed.close();
            log.debugf("Invalidated KV-Cache for session=%s", sessionId);
        }
    }

    /**
     * Clear all caches (e.g. on shutdown).
     */
    public void clearAll() {
        caches.values().forEach(KVCache::close);
        caches.clear();
        log.info("Cleared all KV-Caches");
    }

    /**
     * Number of active caches.
     */
    public int size() {
        return caches.size();
    }

    private void evictOldest() {
        caches.entrySet().stream()
                .min((a, b) -> Long.compare(a.getValue().lastAccessTime(), b.getValue().lastAccessTime()))
                .ifPresent(entry -> {
                    log.debugf("Evicting oldest KV-Cache: session=%s", entry.getKey());
                    invalidate(entry.getKey());
                });
    }

    /**
     * Per-session KV-Cache holding key and value tensors for each layer.
     * <p>
     * The cache grows as tokens are generated. When a new forward pass
     * is executed, only the new token's Q/K/V need to be computed;
     * prior K/V values are reused from this cache.
     */
    public static class KVCache implements AutoCloseable {
        private final int numLayers;
        private final int numHeads;
        private final int headDim;
        private long[][] keyCache; // [layer][seqLen * numHeads * headDim]
        private long[][] valueCache; // [layer][seqLen * numHeads * headDim]
        private int seqLen;
        private volatile long lastAccess;

        KVCache(int numLayers, int numHeads, int headDim) {
            this.numLayers = numLayers;
            this.numHeads = numHeads;
            this.headDim = headDim;
            this.keyCache = new long[numLayers][0];
            this.valueCache = new long[numLayers][0];
            this.seqLen = 0;
            this.lastAccess = System.currentTimeMillis();
        }

        /**
         * Append a step's K/V data for each layer.
         *
         * @param layerKeys   key data per layer (flattened: numHeads * headDim)
         * @param layerValues value data per layer (flattened: numHeads * headDim)
         */
        public void append(float[][] layerKeys, float[][] layerValues) {
            this.lastAccess = System.currentTimeMillis();
            int stepSize = numHeads * headDim;

            for (int layer = 0; layer < numLayers; layer++) {
                keyCache[layer] = appendArray(keyCache[layer], layerKeys[layer], stepSize);
                valueCache[layer] = appendArray(valueCache[layer], layerValues[layer], stepSize);
            }
            seqLen++;
        }

        /**
         * Get the current sequence length (number of cached positions).
         */
        public int getSeqLen() {
            return seqLen;
        }

        /**
         * Get the number of layers in this cache.
         */
        public int getNumLayers() {
            return numLayers;
        }

        public long lastAccessTime() {
            return lastAccess;
        }

        /**
         * Reset cache to empty (e.g., for a new conversation turn).
         */
        public void reset() {
            for (int i = 0; i < numLayers; i++) {
                keyCache[i] = new long[0];
                valueCache[i] = new long[0];
            }
            seqLen = 0;
        }

        @Override
        public void close() {
            reset();
        }

        private long[] appendArray(long[] existing, float[] newData, int stepSize) {
            long[] result = new long[existing.length + stepSize];
            System.arraycopy(existing, 0, result, 0, existing.length);
            for (int i = 0; i < stepSize && i < newData.length; i++) {
                result[existing.length + i] = Float.floatToRawIntBits(newData[i]);
            }
            return result;
        }
    }
}
