package tech.kayys.gollek.runtime.service;

import tech.kayys.alkhawarizm.core.tensor.Tensor;
import tech.kayys.gollek.runtime.kv.*;

/**
 * Prefill Runtime (throughput-optimized):
 * - runs full prompt [T, D]
 * - heavy compute (FlashAttention full)
 * - fills KV cache
 * - can be batched across users
 * - can run on GPU / big CPU node
 *
 * Example:
 * PrefillService prefill = new PrefillService(new FP16Codec());
 * KVCacheSnapshot snapshot = prefill.prefill(
 * promptTensor,
 * wqkv,
 * 32,
 * 2048
 * );
 * // send snapshot over network
 */
public final class PrefillService {
    private final KVCodec codec;

    public PrefillService(KVCodec codec) {
        this.codec = codec;
    }

    public KVCacheSnapshot prefill(
            Tensor prompt, Tensor wqkv,
            int heads,
            int maxSeq) {
        int dim = (int) prompt.shape().dim(1);
        int headDim = dim / heads;
        KVCache cache = new KVCache(maxSeq, heads, headDim, codec);
        // reuse PrefillRunner
        tech.kayys.gollek.runtime.PrefillRunner.run(
                prompt,
                wqkv,
                cache,
                heads,
                prompt.backend());
        return KVCacheSerializer.snapshot(cache);
    }
}