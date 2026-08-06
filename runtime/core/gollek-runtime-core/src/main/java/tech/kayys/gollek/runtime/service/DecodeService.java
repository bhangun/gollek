package tech.kayys.gollek.runtime.service;

import tech.kayys.alkhawarizm.core.tensor.Tensor;
import tech.kayys.gollek.runtime.kv.*;

/**
 * 
 * Decode Runtime (latency-optimized):
 * - runs step-by-step [1, D]
 * - uses KV cache
 * - low latency critical
 * - can run on:
 * - edge
 * - smaller CPU
 * - mobile
 * 
 * Example:
 * 
 * DecodeService decode = new DecodeService(new FP16Codec());
 * KVCache cache = decode.load(snapshot);
 * Tensor out = decode.step(
 * tokenTensor,
 * wqkv,cache,
 * 32
 * );
 */
public final class DecodeService {
    private final KVCodec codec;

    public DecodeService(KVCodec codec) {
        this.codec = codec;
    }

    public KVCache load(KVCacheSnapshot snap) {
        return KVCacheSerializer.restore(snap, codec);
    }

    public Tensor step(Tensor token,
            Tensor wqkv,
            KVCache cache,
            int heads) {
        return tech.kayys.gollek.runtime.DecodeRunner.step(
                token,
                wqkv,
                cache,
                heads,
                token.backend());
    }
}