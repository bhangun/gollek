package tech.kayys.gollek.runtime;

import tech.kayys.alkhawarizm.core.tensor.Tensor;
import tech.kayys.alkhawarizm.core.backend.ComputeBackend;
import tech.kayys.gollek.runtime.kv.KVCache;
import java.util.List;

/**
 * PREFILL (a.k.a. prompt processing)
 * Input: full sequence [T, D]
 * Output: KV cache filled for all tokens
 * Compute: full attention over sequence
 */
public final class PrefillRunner {
    private PrefillRunner() {
    }

    public static Tensor run(
            Tensor x, // [T, D]
            Tensor wqkv,
            KVCache cache,
            int heads,
            ComputeBackend backend) {
        
        long t = x.shape().dim(0);
        int headDim = (int) (x.shape().dim(1) / heads);

        // 1. Compute QKV projection for full sequence
        // [T, D] * [D, 3*D] -> [T, 3*D]
        Tensor qkv = x.matmul(wqkv);

        // 2. Split into Q, K, V
        List<Tensor> parts = backend.split(qkv, 1, 3);
        Tensor q = parts.get(0); // [T, D]
        Tensor k = parts.get(1); // [T, D]
        Tensor v = parts.get(2); // [T, D]

        // 3. Reshape for attention
        q = q.reshape(heads, t, headDim);
        k = k.reshape(heads, t, headDim);
        v = v.reshape(heads, t, headDim);

        // 4. Update KV cache with all tokens
        cache.append(k, v);

        // 5. Compute self-attention for the prefill stage
        // In prefill, we use causal mask internally in backend.attention if T > 1
        return backend.attention(q, k, v);
    }
}
