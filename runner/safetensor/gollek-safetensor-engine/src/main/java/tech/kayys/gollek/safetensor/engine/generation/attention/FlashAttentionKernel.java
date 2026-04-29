package tech.kayys.gollek.safetensor.engine.generation.attention;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import tech.kayys.gollek.metal.MetalComputeBackend;
import tech.kayys.gollek.safetensor.engine.generation.kv.BlockManager;
import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;
import tech.kayys.gollek.safetensor.core.tensor.AccelOps;
import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.spi.model.ModelArchitecture;
import tech.kayys.gollek.spi.model.ModelConfig;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.List;
import jakarta.enterprise.inject.Instance;

@ApplicationScoped
public class FlashAttentionKernel {

    @Inject
    Instance<MetalComputeBackend> metalBackend;
    @Inject
    RopeFrequencyCache ropeCache;

    public static class AttentionInput {
        public final AccelTensor x;
        public final AccelTensor qW, kW, vW, oW;
        public final AccelTensor qB, kB, vB, oB;
        public final ModelArchitecture arch;
        public final ModelConfig config;
        public final KVCacheManager.KVCacheSession kvCache;
        public final int layerIdx;
        public final int startPos;
        public final boolean isCausal;
        public final AccelTensor qNormW, kNormW;
        public final AccelTensor postAttnNormW;

        public AttentionInput(AccelTensor x, AccelTensor qW, AccelTensor kW, AccelTensor vW, AccelTensor oW,
                AccelTensor qB, AccelTensor kB, AccelTensor vB, AccelTensor oB,
                ModelArchitecture arch, ModelConfig config, KVCacheManager.KVCacheSession kvCache,
                int layerIdx, int startPos, boolean isCausal,
                AccelTensor qNormW, AccelTensor kNormW, AccelTensor postAttnNormW) {
            this.x = x;
            this.qW = qW;
            this.kW = kW;
            this.vW = vW;
            this.oW = oW;
            this.qB = qB;
            this.kB = kB;
            this.vB = vB;
            this.oB = oB;
            this.arch = arch;
            this.config = config;
            this.kvCache = kvCache;
            this.layerIdx = layerIdx;
            this.startPos = startPos;
            this.isCausal = isCausal;
            this.qNormW = qNormW;
            this.kNormW = kNormW;
            this.postAttnNormW = postAttnNormW;
        }
    }

    public AccelTensor compute(AttentionInput in) {
        ModelConfig config = in.config;
        int layerIdx = in.layerIdx;
        int startPos = in.startPos;
        int seqLen = (int) in.x.size(1);
        int headDim = config.resolvedMaxHeadDim();
        int numQHeads = config.numAttentionHeads();
        int numKVHeads = config.resolvedNumKvHeads();
        float scale = (float) (1.0 / Math.sqrt(headDim));

        // 1. Projections
        AccelTensor q = AccelOps.linear(in.x, in.qW, in.qB);
        AccelTensor k = AccelOps.linear(in.x, in.kW, in.kB);
        AccelTensor v = AccelOps.linear(in.x, in.vW, in.vB);

        // 2. QK-Norm (Gemma-3/4)
        if (in.qNormW != null) {
            AccelTensor qNormed = rmsNorm(q, in.qNormW, config.rmsNormEps(), in.arch.addOneToRmsNormWeight());
            q.close();
            q = qNormed;
        }
        if (in.kNormW != null) {
            AccelTensor kNormed = rmsNorm(k, in.kNormW, config.rmsNormEps(), in.arch.addOneToRmsNormWeight());
            k.close();
            k = kNormed;
        }

        // 3. RoPE
        RopeFrequencyCache.RopeFrequencies freqs = ropeCache.get(headDim, config.maxPositionEmbeddings(), 10000.0,
                config.ropeScaling());
        applyRope(q, k, startPos, freqs, !in.arch.usesNeoxRope());

        // 4. Update Cache
        KVCacheManager.KVCacheSession kvSession = in.kvCache;
        updateKVCache(k, v, kvSession, layerIdx, startPos, seqLen, numKVHeads, headDim);

        // 5. Attention
        AccelTensor attnOut;
        if (metalBackend.isResolvable() && metalBackend.get().deviceName() != null
                && !metalBackend.get().deviceName().contains("CPU")) {
            attnOut = tiledAttention(q, kvSession, layerIdx, startPos, numQHeads, headDim, scale, in.isCausal,
                    in.arch.defaultAttnSoftCap());
        } else {
            attnOut = PagedAttentionVectorAPI.compute(q, kvSession, layerIdx, startPos, numQHeads, headDim, scale,
                    in.isCausal, in.arch.defaultAttnSoftCap());
        }
        q.close();
        k.close();
        v.close();

        // 6. Post-Attention Norm (Gemma-2/4)
        AccelTensor projectedIn = attnOut;
        if (in.postAttnNormW != null) {
            projectedIn = rmsNorm(attnOut, in.postAttnNormW, config.rmsNormEps(), in.arch.addOneToRmsNormWeight());
            attnOut.close();
        }

        // 7. Output Projection
        AccelTensor out = AccelOps.linear(projectedIn, in.oW, in.oB);
        projectedIn.close();

        return out;
    }

    private AccelTensor rmsNorm(AccelTensor x, AccelTensor w, double eps, boolean addOne) {
        return AccelOps.rmsNorm(x, w, eps, addOne);
    }

    private void applyRope(AccelTensor q, AccelTensor k, int startPos, RopeFrequencyCache.RopeFrequencies freqs,
            boolean interleaved) {
        int seqLen = (int) q.size(1);
        int numQHeads = (int) q.size(2);
        int numKVHeads = (int) k.size(2);
        int headDim = (int) q.size(3);

        for (int s = 0; s < seqLen; s++) {
            int pos = startPos + s;
            for (int h = 0; h < numQHeads; h++) {
                freqs.rotateInPlace(q.dataSegment(), ((long) s * numQHeads + h) * headDim, pos, interleaved);
            }
            for (int h = 0; h < numKVHeads; h++) {
                freqs.rotateInPlace(k.dataSegment(), ((long) s * numKVHeads + h) * headDim, pos, interleaved);
            }
        }
    }

    private void updateKVCache(AccelTensor k, AccelTensor v, KVCacheManager.KVCacheSession kvSession, int layerIdx,
            int startPos, int seqLen, int numHeads, int headDim) {
        BlockManager blockManager = kvSession.blockManager();
        MemorySegment kSeg = k.dataSegment();
        MemorySegment vSeg = v.dataSegment();

        int tokensPerBlock = kvSession.tokensPerBlock();
        long headStride = blockManager.getHeadStride();
        long tokenStride = blockManager.getTokenStride();

        for (int s = 0; s < seqLen; s++) {
            int absolutePos = startPos + s;
            int blockIdx = kvSession.getBlockForToken(layerIdx, absolutePos);
            int tokenIdxInBlock = absolutePos % tokensPerBlock;

            MemorySegment kBlock = blockManager.getKBlock(blockIdx);
            MemorySegment vBlock = blockManager.getVBlock(blockIdx);

            for (int h = 0; h < numHeads; h++) {
                long srcOff = ((long) s * numHeads + h) * headDim;
                long hDstOff = ((long) h * headStride + (long) tokenIdxInBlock * tokenStride);

                MemorySegment.copy(kSeg, ValueLayout.JAVA_FLOAT, srcOff * 4, kBlock, ValueLayout.JAVA_FLOAT,
                        hDstOff * 4, headDim);
                MemorySegment.copy(vSeg, ValueLayout.JAVA_FLOAT, srcOff * 4, vBlock, ValueLayout.JAVA_FLOAT,
                        hDstOff * 4, headDim);
            }
        }
    }

    private AccelTensor tiledAttention(AccelTensor q, KVCacheManager.KVCacheSession kvSession, int layerIdx,
            int startPos, int numHeads, int headDim, float scale, boolean causal, float softCap) {
        BlockManager blockManager = kvSession.blockManager();
        long batch = q.size(0);
        long seqLen = q.size(1);
        int totalTokens = startPos + (int) seqLen;

        try (Arena arena = Arena.ofConfined()) {
            List<Integer> blocks = kvSession.getBlockIndices(layerIdx);
            MemorySegment btSeg = arena.allocate(ValueLayout.JAVA_INT, blocks.size());
            for (int i = 0; i < blocks.size(); i++)
                btSeg.setAtIndex(ValueLayout.JAVA_INT, i, blocks.get(i));

            MemorySegment clSeg = arena.allocate(ValueLayout.JAVA_INT, batch);
            for (int i = 0; i < batch; i++)
                clSeg.setAtIndex(ValueLayout.JAVA_INT, i, totalTokens);

            AccelTensor out = AccelTensor.zeros(q.shape());
            metalBackend.get().pagedAttention(
                    out.dataSegment(), q.dataSegment(),
                    blockManager.getRawKPool(), blockManager.getRawVPool(),
                    btSeg, clSeg,
                    (int) batch, (int) seqLen, numHeads, headDim,
                    kvSession.tokensPerBlock(), blocks.size(),
                    scale, causal, softCap);
            return out;
        }
    }
}
