package tech.kayys.gollek.ml.inference;

import tech.kayys.gollek.ml.autograd.GradTensor;
import tech.kayys.gollek.ml.tensor.VectorOps;

import java.lang.foreign.*;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * KV Cache — caches key and value tensors across autoregressive generation steps
 * to avoid recomputing attention for previously seen tokens.
 *
 * <p>Without KV cache: each step recomputes attention over all T tokens → O(T²) total.
 * With KV cache: each step only computes attention for the new token → O(T) total.
 *
 * <p>Uses JDK 25 FFM {@link MemorySegment} for off-heap storage to avoid GC pressure
 * during long generation sequences.
 *
 * <h3>Example</h3>
 * <pre>{@code
 * var cache = new KVCache(nLayers=12, nHeads=12, headDim=64, maxSeqLen=2048);
 *
 * for (int step = 0; step < maxNewTokens; step++) {
 *     // Only pass the new token, not the full sequence
 *     GradTensor k = computeKey(newToken);
 *     GradTensor v = computeValue(newToken);
 *     cache.update(layer=0, k, v);
 *
 *     // Retrieve full K, V for attention
 *     GradTensor fullK = cache.getKeys(layer=0);
 *     GradTensor fullV = cache.getValues(layer=0);
 * }
 *
 * cache.reset(); // clear for next generation
 * }</pre>
 */
public final class KVCache implements AutoCloseable {

    private final int nLayers;
    private final int nHeads;
    private final int headDim;
    private final int maxSeqLen;

    /** Off-heap storage: [nLayers][2 (K,V)][nHeads * maxSeqLen * headDim] */
    private final MemorySegment[] keySegments;
    private final MemorySegment[] valSegments;
    private final Arena arena;
    private final int[] seqLens; // current sequence length per layer

    /**
     * Creates a KV cache with off-heap storage.
     *
     * @param nLayers   number of transformer layers
     * @param nHeads    number of attention heads
     * @param headDim   dimension per head
     * @param maxSeqLen maximum sequence length to cache
     */
    public KVCache(int nLayers, int nHeads, int headDim, int maxSeqLen) {
        this.nLayers   = nLayers;
        this.nHeads    = nHeads;
        this.headDim   = headDim;
        this.maxSeqLen = maxSeqLen;
        this.seqLens   = new int[nLayers];
        this.arena     = Arena.ofShared();

        long slotBytes = (long) nHeads * maxSeqLen * headDim * Float.BYTES;
        this.keySegments = new MemorySegment[nLayers];
        this.valSegments = new MemorySegment[nLayers];
        for (int l = 0; l < nLayers; l++) {
            keySegments[l] = arena.allocate(slotBytes, Float.BYTES);
            valSegments[l] = arena.allocate(slotBytes, Float.BYTES);
        }
    }

    /**
     * Appends new key and value tensors for the current token at the given layer.
     *
     * @param layer layer index (0-indexed)
     * @param k     new key tensor {@code [nHeads, 1, headDim]}
     * @param v     new value tensor {@code [nHeads, 1, headDim]}
     * @throws IllegalStateException if the cache is full
     */
    public void update(int layer, GradTensor k, GradTensor v) {
        if (seqLens[layer] >= maxSeqLen)
            throw new IllegalStateException("KV cache full at layer " + layer);

        int pos = seqLens[layer];
        long offset = (long) pos * nHeads * headDim * Float.BYTES;
        long len    = (long) nHeads * headDim * Float.BYTES;

        MemorySegment.copy(MemorySegment.ofArray(k.data()), 0, keySegments[layer], offset, len);
        MemorySegment.copy(MemorySegment.ofArray(v.data()), 0, valSegments[layer], offset, len);
        seqLens[layer]++;
    }

    /**
     * Returns all cached keys for the given layer as a tensor.
     *
     * @param layer layer index
     * @return key tensor {@code [nHeads, seqLen, headDim]}
     */
    public GradTensor getKeys(int layer) {
        return readCache(keySegments[layer], layer);
    }

    /**
     * Returns all cached values for the given layer as a tensor.
     *
     * @param layer layer index
     * @return value tensor {@code [nHeads, seqLen, headDim]}
     */
    public GradTensor getValues(int layer) {
        return readCache(valSegments[layer], layer);
    }

    /**
     * Returns the current sequence length (number of cached tokens) for a layer.
     *
     * @param layer layer index
     * @return number of cached tokens
     */
    public int seqLen(int layer) { return seqLens[layer]; }

    /**
     * Resets the cache for all layers (clears sequence lengths, data remains but is overwritten).
     */
    public void reset() { java.util.Arrays.fill(seqLens, 0); }

    /**
     * Resets the cache for a specific layer.
     *
     * @param layer layer index
     */
    public void reset(int layer) { seqLens[layer] = 0; }

    @Override
    public void close() { arena.close(); }

    private GradTensor readCache(MemorySegment seg, int layer) {
        int T = seqLens[layer];
        float[] data = new float[nHeads * T * headDim];
        MemorySegment.copy(seg, 0, MemorySegment.ofArray(data), 0,
            (long) nHeads * T * headDim * Float.BYTES);
        return GradTensor.of(data, nHeads, T, headDim);
    }
}
