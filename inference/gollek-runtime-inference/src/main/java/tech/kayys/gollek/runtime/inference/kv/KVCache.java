package tech.kayys.gollek.runtime.inference.kv;

import tech.kayys.gollek.runtime.tensor.Tensor;

/**
 * Key-Value cache for transformer attention layers.
 * <p>
 * Stores past K and V projections per layer so that each new token
 * only computes its own K/V and appends to the cache, avoiding the
 * O(n²) recomputation problem.
 * <p>
 * Implementations include:
 * <ul>
 *   <li>{@link PagedKVCache} — vLLM-style paged memory</li>
 * </ul>
 */
public interface KVCache {

    /**
     * Append new K and V tensors for the given layer.
     *
     * @param layer attention layer index
     * @param k     key projection for the new token(s)
     * @param v     value projection for the new token(s)
     */
    void append(int layer, Tensor k, Tensor v);

    /** Get the full cached K tensor for the given layer. */
    Tensor getK(int layer);

    /** Get the full cached V tensor for the given layer. */
    Tensor getV(int layer);

    /** Total number of cached tokens across all layers. */
    int length();

    /** Clear all cached state (e.g. between requests). */
    void clear();

    /**
     * Create a snapshot of the current cache state.
     * Used by prefix caching to save and restore shared prefixes.
     */
    KVCache snapshot();
}
