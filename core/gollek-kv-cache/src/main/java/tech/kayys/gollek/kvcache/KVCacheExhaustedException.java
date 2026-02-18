package tech.kayys.gollek.kvcache;

/**
 * Thrown when the KV-Cache block pool is exhausted and cannot satisfy
 * an allocation request.
 * <p>
 * Recovery options:
 * <ul>
 *   <li>Evict idle/completed requests via {@link PagedKVCacheManager#freeRequest(String)}</li>
 *   <li>Increase {@code totalBlocks} in {@link KVCacheConfig}</li>
 *   <li>Reject the incoming request (backpressure)</li>
 *   <li>Queue the request until blocks become available</li>
 * </ul>
 */
public class KVCacheExhaustedException extends RuntimeException {

    public KVCacheExhaustedException(String message) {
        super(message);
    }

    public KVCacheExhaustedException(String message, Throwable cause) {
        super(message, cause);
    }
}
