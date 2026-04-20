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
import tech.kayys.gollek.safetensor.engine.generation.kv.BlockManager;
import tech.kayys.gollek.safetensor.mask.CausalMaskKernel;
import jakarta.inject.Inject;
import java.util.List;
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

    @Inject
    BlockManager blockManager;

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
            // Compute Multi-Head Attention
            AccelTensor attnOut = tiledAttention(qReshaped, in.kvCache(), scale, in.isCausal(), config, in.startPos(), in.layerIdx());

            // Merge heads: [B, H, S, D] -> [B, S, H, D] -> [B, S, Hidden]
            AccelTensor attnTrans = attnOut.transpose(1, 2).contiguous();
            AccelTensor attnFlat = attnTrans.reshape(batch, seqLen, (long) numQHeads * headDim);

            AccelTensor proj = linear(attnFlat, in.oWeight(), in.oBias());

            // Cleanup local tensors
            qReshaped.close(); kReshaped.close(); vReshaped.close();
            attnOut.close(); attnTrans.close(); attnFlat.close();

            return proj;
        } finally {
            q.close(); k.close(); v.close();
        }
    }

    private AccelTensor tiledAttention(AccelTensor q, KVCacheManager.KVCacheSession kvSession, float scale, boolean causal, ModelConfig config, int startPos, int layerIdx) {
        List<Integer> blockTable = kvSession.getBlockTable(layerIdx);
        int totalTokens = startPos + (int) q.size(2);
        
        return PagedAttentionVectorAPI.compute(
                q, 
                blockTable, 
                blockManager, 
                scale, 
                causal, 
                kvSession.tokensPerBlock(), 
                totalTokens);
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
        int numHeads = (int) k.size(1);
        int headDim = (int) k.size(3);
        int tokensPerBlock = cache.tokensPerBlock();
        List<Integer> blockTable = cache.getBlockTable(layer);

        MemorySegment kSeg = k.dataSegment();
        MemorySegment vSeg = v.dataSegment();

        // Write each token into its corresponding block
        for (int s = 0; s < seqLen; s++) {
            int absPos = startPos + s;
            int blockIdxInTable = absPos / tokensPerBlock;
            int tokenIdxInBlock = absPos % tokensPerBlock;
            
            int physicalBlock = blockTable.get(blockIdxInTable);
            MemorySegment blockSeg = blockManager.getBlock(physicalBlock);

            // Layout in PagedAttentionVectorAPI: [tokensPerBlock, numHeads, headDim, 2 (K, V)]
            long headStride = (long) headDim * 2;
            long tokStride = (long) numHeads * headStride;

            for (int h = 0; h < numHeads; h++) {
                long srcOff = ((long) h * seqLen + s) * headDim;
                long dstOffBase = ((long) tokenIdxInBlock * tokStride + (long) h * headStride);
                
                // Copy K
                MemorySegment.copy(kSeg, ValueLayout.JAVA_FLOAT, srcOff * 4, blockSeg, ValueLayout.JAVA_FLOAT, dstOffBase * 4, headDim);
                // Copy V
                MemorySegment.copy(vSeg, ValueLayout.JAVA_FLOAT, srcOff * 4, blockSeg, ValueLayout.JAVA_FLOAT, (dstOffBase + headDim) * 4, headDim);
            }
        }
    }


    private AccelTensor repeatKV(AccelTensor kv, int groupSize) {
        long[] sh = kv.shape();
        int b = (int) sh[0], numKVHeads = (int) sh[1], seqLen = (int) sh[2], headDim = (int) sh[3];
        int numQHeads = numKVHeads * groupSize;
        
        AccelTensor exp = AccelTensor.zeros(b, numQHeads, seqLen, headDim);
        MemorySegment src = kv.dataSegment();
        MemorySegment dst = exp.dataSegment();
        
        // Optimized repeatKV using bulk MemorySegment copies for [B, H, S, D]
        for (int i = 0; i < b; i++) {
            for (int hk = 0; hk < numKVHeads; hk++) {
                for (int g = 0; g < groupSize; g++) {
                    int hq = hk * groupSize + g;
                    long srcOff = ((long) i * numKVHeads + hk) * seqLen * headDim;
                    long dstOff = ((long) i * numQHeads + hq) * seqLen * headDim;
                    MemorySegment.copy(src, ValueLayout.JAVA_FLOAT, srcOff * ValueLayout.JAVA_FLOAT.byteSize(), dst, ValueLayout.JAVA_FLOAT, dstOff * ValueLayout.JAVA_FLOAT.byteSize(), (long) seqLen * headDim);
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
