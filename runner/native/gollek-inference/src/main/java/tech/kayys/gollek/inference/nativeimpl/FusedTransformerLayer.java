package tech.kayys.gollek.inference.nativeimpl;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.concurrent.ExecutorService;
import tech.kayys.gollek.gguf.loader.TransformerLayerWeights;

/**
 * Executes a single transformer layer end-to-end with zero intermediary allocations.
 */
public final class FusedTransformerLayer {

    private FusedTransformerLayer() {}

    public static void execute(
        int layerIdx,
        int pos,
        MemorySegment x,
        MemorySegment xNext,
        TransformerLayerWeights w,
        LayerBuffers buf,
        KVCache kvCache,
        RoPECache ropeCache,
        int hidden,
        int numHeads,
        int numHeadsKv,
        int headDim,
        boolean isNeox,
        float eps,
        float attnSoftCap,
        ExecutorService executor
    ) {
        // 1. RMSNorm (Attention) -> buf.norm
        RMSNormKernel.execute(x, buf.norm, w.rmsWeight, hidden, eps);
        
        // 2. QKV Projection -> buf.q, buf.k, buf.v (handles QKV biases)
        PrepackedQKVKernel.computeParallel(buf.norm, w.wqkvPacked, w.bqkv, buf.q, buf.k, buf.v, hidden, numHeads, numHeadsKv, headDim, executor);

        // 3. Apply RoPE
        for (int h = 0; h < numHeads; h++) {
            MemorySegment qHead = buf.q.asSlice((long) h * headDim * Float.BYTES, (long) headDim * Float.BYTES);
            RoPEKernel.apply(qHead, pos, ropeCache, headDim, isNeox);
        }
        for (int h = 0; h < numHeadsKv; h++) {
            MemorySegment kHead = buf.k.asSlice((long) h * headDim * Float.BYTES, (long) headDim * Float.BYTES);
            RoPEKernel.apply(kHead, pos, ropeCache, headDim, isNeox);
        }

        // 4. Update KV Cache
        int qHeadsPerKvHead = numHeads / numHeadsKv;
        for (int h = 0; h < numHeadsKv; h++) {
            for (int d = 0; d < headDim; d++) {
                float kval = buf.k.get(ValueLayout.JAVA_FLOAT, (long)((long) h * headDim + d) * Float.BYTES);
                float vval = buf.v.get(ValueLayout.JAVA_FLOAT, (long)((long) h * headDim + d) * Float.BYTES);
                // We use the first Q head of the group as the storage key
                kvCache.setK(layerIdx, h * qHeadsPerKvHead, pos, d, kval);
                kvCache.setV(layerIdx, h * qHeadsPerKvHead, pos, d, vval);
            }
        }

        // 5. FlashAttention -> buf.attnOut
        FlashAttentionKernel.computeParallel(buf.attnOut, buf.q, kvCache, null, layerIdx, pos + 1, numHeads, headDim, attnSoftCap, executor);

        // 6. Output Projection and Residual connection (handles Output bias)
        LinearResidualKernel.computeParallel(buf.attnOut, w.wo, w.bo, x, x, hidden, hidden, executor);

        // 7. RMSNorm (FFN) -> buf.norm
        RMSNormKernel.execute(x, buf.norm, w.ffnNormWeight, hidden, eps);

        // 8. FFN & Residual -> xNext (final layer output for this step, handles Gate/Up/Down biases)
        int ffnDim = (int) (buf.ffn.byteSize() / Float.BYTES);
        FusedFFNKernel.computeParallel(buf.norm, w, buf.ffn, x, xNext, hidden, ffnDim, executor);
    }
}
