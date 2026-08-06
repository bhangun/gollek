package tech.kayys.gollek.runtime;

import tech.kayys.alkhawarizm.core.tensor.Tensor;
import tech.kayys.alkhawarizm.core.backend.ComputeBackend;
import tech.kayys.gollek.runtime.kv.KVCache;
import java.util.List;

/**
 * DECODE (a.k.a. autoregressive):
 * Input: single token [1, D]
 * Output: next token
 * Compute: attention over KV cache
 */
public final class DecodeRunner {
    private DecodeRunner() {
    }

    public static Tensor step(
            Tensor x, // [1, D]
            Tensor wqkv,
            KVCache cache,
            int heads,
            ComputeBackend backend) {
        
        // 1. Compute QKV projection
        // x is [1, D], wqkv is [D, 3*D] (assuming MHA)
        // result qkv is [1, 3*D]
        Tensor qkv = x.matmul(wqkv);

        // 2. Split into Q, K, V
        // Assuming concatenated Q, K, V along the last dimension
        List<Tensor> parts = backend.split(qkv, 1, 3);
        Tensor q = parts.get(0); // [1, D]
        Tensor k = parts.get(1); // [1, D]
        Tensor v = parts.get(2); // [1, D]

        // 3. Reshape for attention: [heads, 1, headDim]
        int headDim = (int) (x.shape().dim(1) / heads);
        q = q.reshape(heads, 1, headDim);
        k = k.reshape(heads, 1, headDim);
        v = v.reshape(heads, 1, headDim);

        // 4. Update KV cache
        cache.append(k, v);

        // 5. Get full KV for attention
        Tensor fullK = cache.getFullK(backend); // [pos, heads, headDim]
        Tensor fullV = cache.getFullV(backend); // [pos, heads, headDim]

        // 6. Compute attention
        // Note: fullK/fullV might need reshape to match [heads, pos, headDim] 
        // depending on backend.attention expectation.
        Tensor attnOut = backend.attention(q, fullK, fullV);

        // 7. Final projection (wo) would usually happen after this, 
        // but it's often outside the core attention runner.
        return attnOut;
    }
}
