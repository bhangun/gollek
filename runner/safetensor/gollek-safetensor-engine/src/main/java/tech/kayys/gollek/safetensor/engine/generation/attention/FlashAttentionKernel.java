/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;
import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.core.tensor.AccelOps;
import tech.kayys.gollek.spi.model.ModelConfig;
import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;
import tech.kayys.gollek.safetensor.mask.CausalMaskKernel;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

/**
 * Attention kernel using AccelTensor + Apple Accelerate.
 * No LibTorch dependency.
 */
@ApplicationScoped
public class FlashAttentionKernel {

    private static final Logger log = Logger.getLogger(FlashAttentionKernel.class);

    @Inject
    RopeFrequencyCache ropeCache;

    public record AttentionInput(
            AccelTensor hiddenState,
            AccelTensor qWeight, AccelTensor kWeight, AccelTensor vWeight, AccelTensor oWeight,
            AccelTensor qBias, AccelTensor kBias, AccelTensor vBias, AccelTensor oBias,
            ModelConfig config,
            KVCacheManager.KVCacheSession kvCache,
            int layerIdx,
            int startPos,
            boolean isCausal,
            AccelTensor qNormWeight,
            AccelTensor kNormWeight,
            AccelTensor postAttnNorm
    ) {
    }

    public AccelTensor compute(AttentionInput in) {
        ModelConfig config = in.config();
        int numQHeads = config.numAttentionHeads();
        int numKVHeads = config.resolvedNumKvHeads();
        int headDim = config.resolvedHeadDim();
        int groupSize = numKVHeads > 0 ? (numQHeads / numKVHeads) : 1;
        float scale = (float) (1.0 / Math.sqrt(headDim));

        // Projections
        AccelTensor q = linear(in.hiddenState(), in.qWeight(), in.qBias());
        AccelTensor k = linear(in.hiddenState(), in.kWeight(), in.kBias());
        AccelTensor v = linear(in.hiddenState(), in.vWeight(), in.vBias());

        try {
            long batch = in.hiddenState().size(0);
            long seqLen = in.hiddenState().size(1);

            // Attention Norms (Q-Norm / K-Norm) - Critical for Qwen 2.5
            if (in.qNormWeight() != null) {
                AccelTensor qNormed = AccelOps.rmsNorm(q, in.qNormWeight(), config.rmsNormEps());
                q.close();
                q = qNormed;
            }
            if (in.kNormWeight() != null) {
                AccelTensor kNormed = AccelOps.rmsNorm(k, in.kNormWeight(), config.rmsNormEps());
                k.close();
                k = kNormed;
            }

            AccelTensor qReshaped = q.reshape(batch, (int)seqLen, numQHeads, headDim).transpose(1, 2).contiguous();
            AccelTensor kReshaped = k.reshape(batch, (int)seqLen, numKVHeads, headDim).transpose(1, 2).contiguous();
            AccelTensor vReshaped = v.reshape(batch, (int)seqLen, numKVHeads, headDim).transpose(1, 2).contiguous();

            // RoPE
            double theta = config.ropeThetaForLayer(in.layerIdx());
            int rotaryDim = (int) (headDim * config.partialRotaryFactorForLayer(in.layerIdx()));
            if (rotaryDim > 0) {
                RopeFrequencyCache.RopeFrequencies rope = ropeCache.get(rotaryDim, config.maxPositionEmbeddings(), theta);
                
                // Optimized RoPE: Process in-place on continuous segments
                applyRopeToSegment(qReshaped.dataSegment(), (int)batch, numQHeads, (int)seqLen, headDim, in.startPos(), rope);
                applyRopeToSegment(kReshaped.dataSegment(), (int)batch, numKVHeads, (int)seqLen, headDim, in.startPos(), rope);
            }

            // KV Cache: Update with current block [B, H, S, D]
            updateKVCache(in.kvCache(), in.layerIdx(), qReshaped, kReshaped, vReshaped, in.startPos());

            // Read full context views: [B, H, cacheLen, D]
            int cacheLen = in.startPos() + (int) seqLen;
            AccelTensor kView = readKVCache(in.kvCache(), in.layerIdx(), (int) batch, numKVHeads, headDim, cacheLen, true);
            AccelTensor vView = readKVCache(in.kvCache(), in.layerIdx(), (int) batch, numKVHeads, headDim, cacheLen, false);

            // Compute Multi-Head Attention
            AccelTensor attnOut = tiledAttention(qReshaped, kView, vView, scale, in.isCausal(), config, in.startPos(), in.layerIdx());

            // Merge heads: [B, H, S, D] -> [B, S, H, D] -> [B, S, Hidden]
            AccelTensor attnTrans = attnOut.transpose(1, 2).contiguous();
            AccelTensor attnFlat = attnTrans.reshape(batch, seqLen, (long) numQHeads * headDim);

            AccelTensor proj = linear(attnFlat, in.oWeight(), in.oBias());

            // Cleanup local tensors
            qReshaped.close(); kReshaped.close(); vReshaped.close();
            kView.close(); vView.close();
            attnOut.close(); attnTrans.close(); attnFlat.close();

            return proj;
        } finally {
            q.close(); k.close(); v.close();
        }
    }

    private AccelTensor tiledAttention(AccelTensor q, AccelTensor k, AccelTensor v, float scale, boolean causal, ModelConfig config, int startPos, int layerIdx) {
        // Q @ K^T
        AccelTensor kT = k.transpose(2, 3); // Already contiguous from transpose()
        AccelTensor scores = AccelOps.matmul(q, kT);
        kT.close();

        // Scale
        AccelTensor scaled = AccelOps.mulScalar(scores, scale);
        scores.close();

        if (scaled.shape().length == 4 && scaled.size(0) == 1 && scaled.size(2) == 1) {
             // For decoder steps or single-token prefill
        } else if (scaled.shape().length == 4) {
             // For layered diagnostics
        }

        // Causal mask application (ONLY for prompt processing / Prefill)
        if (causal && scaled.size(2) > 1) {
            long[] sh = scaled.shape();
            CausalMaskKernel.applyCausalMask(scaled.dataSegment(), (int)sh[0], (int)sh[1], (int)sh[2], (int)sh[3], startPos);
        }

        // Softmax
        if (layerIdx == 0 && (startPos == 0 || (startPos % 10 == 0))) {
            System.err.println("[DIAG-KV] L" + layerIdx + "_Attention_Scores_Pre_Softmax (pos " + startPos + "): " + scaled.statistics());
        }
        AccelTensor weights = AccelOps.softmax(scaled, 3);
        scaled.close();

        if (layerIdx == 0 && (startPos == 0 || (startPos % 10 == 0))) {
             System.err.println("[DIAG-KV] L" + layerIdx + "_Attn_Weights_Softmax (pos " + startPos + "): " + weights.statistics());
        }

        // Weights @ V
        AccelTensor out = AccelOps.matmul(weights, v);
        weights.close();
        return out;
    }

    private void applyRopeToSegment(MemorySegment seg, int batch, int numHeads, int seqLen, int headDim, int startPos, RopeFrequencyCache.RopeFrequencies rope) {
        for (int b = 0; b < batch; b++) {
            for (int h = 0; h < numHeads; h++) {
                for (int s = 0; s < seqLen; s++) {
                    int pos = startPos + s;
                    // Correct layout: [batch, heads, seq, dim]
                    // Index = (b * numHeads * seqLen * headDim) + (h * seqLen * headDim) + (s * headDim)
                    long elementOffset = (((long)b * numHeads + h) * seqLen + s) * headDim;
                    rope.rotateInPlace(seg, elementOffset, pos);
                }
            }
        }
    }

    private void updateKVCache(KVCacheManager.KVCacheSession cache, int layer, AccelTensor q, AccelTensor k, AccelTensor v, int startPos) {
        int seqLen = (int) k.size(2); // shape [1, H, S, D]
        AccelTensor ks = cache.keyCache(layer);
        AccelTensor vs = cache.valueCache(layer);

        int numHeads = (int) k.size(1);
        int headDim = (int) k.size(3);
        
        MemorySegment kSeg = k.dataSegment();
        MemorySegment vSeg = v.dataSegment();
        MemorySegment ksSeg = ks.dataSegment();
        MemorySegment vsSeg = vs.dataSegment();

        // Target cache layout: [maxSeqLen, numHeads, headDim]
        long cacheStride0 = (long) numHeads * headDim;

        for (int h = 0; h < numHeads; h++) {
            for (int s = 0; s < seqLen; s++) {
                int pos = startPos + s;
                // Source is [1, H, S, D] -> Contiguous dim block
                long srcOff = ((long) h * seqLen + s) * headDim;
                long dstOff = (long) pos * cacheStride0 + (long) h * headDim;
                
                MemorySegment.copy(kSeg, ValueLayout.JAVA_FLOAT, srcOff, ksSeg, ValueLayout.JAVA_FLOAT, dstOff, headDim);
                MemorySegment.copy(vSeg, ValueLayout.JAVA_FLOAT, srcOff, vsSeg, ValueLayout.JAVA_FLOAT, dstOff, headDim);
            }
        }
    }

    private AccelTensor readKVCache(KVCacheManager.KVCacheSession cache, int layer, int batch, int numHeads, int headDim, int len, boolean isKey) {
        AccelTensor s = isKey ? cache.keyCache(layer) : cache.valueCache(layer);
        // physical cache s is [maxSeqLen, numHeads, headDim]
        // return logical view [batch, numHeads, len, headDim]
        // To do this strictly correctly with Zero-Copy, we need to transpose the slice.
        // For now, let's materialize it to be safe.
        AccelTensor out = AccelTensor.zeros(batch, numHeads, len, headDim);
        MemorySegment srcSeg = s.dataSegment();
        MemorySegment dstSeg = out.dataSegment();
        
        long srcStride0 = (long) numHeads * headDim;
        for (int h = 0; h < numHeads; h++) {
            for (int i = 0; i < len; i++) {
                long srcOff = (long) i * srcStride0 + (long) h * headDim;
                long dstOff = ((long) h * len + i) * headDim;
                MemorySegment.copy(srcSeg, ValueLayout.JAVA_FLOAT, srcOff, dstSeg, ValueLayout.JAVA_FLOAT, dstOff, headDim);
            }
        }
        return out;
    }

    private AccelTensor repeatKV(AccelTensor kv, int groupSize) {
        long[] sh = kv.shape();
        int b = (int) sh[0], s = (int) sh[1], nkv = (int) sh[2], dim = (int) sh[3];
        int nq = nkv * groupSize;
        
        AccelTensor exp = AccelTensor.zeros(b, s, nq, dim);
        MemorySegment src = kv.dataSegment();
        MemorySegment dst = exp.dataSegment();
        
        // Optimized repeatKV using bulk MemorySegment copies for [B, H, S, D]
        for (int i = 0; i < b; i++) {
            for (int k = 0; k < nkv; k++) {
                for (int g = 0; g < groupSize; g++) {
                    int hq = k * groupSize + g;
                    long srcOff = ((long) i * nkv + k) * s * dim;
                    long dstOff = ((long) i * nq + hq) * s * dim;
                    MemorySegment.copy(src, ValueLayout.JAVA_FLOAT, srcOff, dst, ValueLayout.JAVA_FLOAT, dstOff, (long) s * dim);
                }
            }
        }
        return exp;
    }

    private AccelTensor linear(AccelTensor x, AccelTensor w, AccelTensor b) {
        AccelTensor out = AccelOps.linear(x, w);
        if (b != null) {
            AccelTensor biased = AccelOps.add(out, b);
            out.close();
            return biased;
        }
        return out;
    }
}
