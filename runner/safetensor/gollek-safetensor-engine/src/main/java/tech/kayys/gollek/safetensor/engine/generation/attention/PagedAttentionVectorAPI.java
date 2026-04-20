package tech.kayys.gollek.safetensor.engine.generation.attention;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.kv.BlockManager;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.List;

/**
 * Paged Attention-2 implementation using Java Vector API.
 * Traverses a block-mapped KV cache for memory-efficient inference.
 */
public class PagedAttentionVectorAPI {

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    /**
     * Compute attention using the Paged Attention algorithm.
     */
    public static AccelTensor compute(
            AccelTensor q, 
            List<Integer> blockTable, 
            BlockManager blockManager,
            float scale, 
            boolean causal,
            int tokensPerBlock,
            int totalTokens) {
            
        long batch = q.size(0);
        long numQHeads = q.size(1);
        long seqLenQ = q.size(2);
        long headDim = q.size(3);
        
        // Assuming numKVHeads = numQHeads for simplicity in this fallback, 
        // but real GQA requires a mapping.
        
        AccelTensor out = AccelTensor.zeros(batch, numQHeads, seqLenQ, headDim);

        for (int b = 0; b < batch; b++) {
            for (int h = 0; h < numQHeads; h++) {
                for (int i = 0; i < seqLenQ; i++) {
                    computePagedHeadQuery(b, h, i, q, blockTable, blockManager, out, scale, causal, tokensPerBlock, totalTokens);
                }
            }
        }
        return out;
    }

    private static void computePagedHeadQuery(int b, int h, int i, 
                                            AccelTensor q, List<Integer> blockTable, BlockManager blockManager, AccelTensor out,
                                            float scale, boolean causal, int tokensPerBlock, int totalTokens) {
        
        MemorySegment qSeg = q.dataSegment();
        MemorySegment oSeg = out.dataSegment();

        long headDim = q.size(3);
        long qOff = ((((long)b * q.size(1) + h) * q.size(2) + i) * headDim) * 4;
        
        float m = Float.NEGATIVE_INFINITY;
        float l = 0.0f;
        float[] acc = new float[(int) headDim];

        // Softmax Online Loop over blocks
        int currentTokCount = 0;
        for (int blockIdx : blockTable) {
            MemorySegment blockSeg = blockManager.getBlock(blockIdx);
            int tokensInThisBlock = Math.min(tokensPerBlock, totalTokens - currentTokCount);
            
            // In each block, K and V are stored back-to-back or interleaved.
            // Layout in BlockManager.init: [tokensPerBlock, numHeads, headDim, 2 (K, V)]
            // Let's use: K_offset = headDim * 0, V_offset = headDim * 1
            long headStride = (long) headDim * 2;
            long tokStride = (long) q.size(1) * headStride;

            for (int tok = 0; tok < tokensInThisBlock; tok++) {
                int absPos = currentTokCount + tok;
                if (causal && absPos > i) break;

                // 1. Compute Dot Product Q_i @ K_j
                long kOff = (long) tok * tokStride + (long) h * headStride;
                float score = dotProduct(qSeg, qOff, blockSeg, kOff * 4, headDim) * scale;

                // 2. Online Softmax Update
                float m_prev = m;
                m = Math.max(m, score);
                float exp_prev = (float) Math.exp(m_prev - m);
                float exp_curr = (float) Math.exp(score - m);
                l = l * exp_prev + exp_curr;

                // 3. Update Output Accumulator
                long vOff = kOff + headDim; // V is right after K
                updateAccumulator(acc, blockSeg, vOff * 4, exp_prev, exp_curr, headDim);
            }
            
            currentTokCount += tokensInThisBlock;
            if (currentTokCount >= totalTokens) break;
        }

        // 4. Final Normalization
        long oOff = ((((long)b * out.size(1) + h) * out.size(2) + i) * headDim) * 4;
        for (int d = 0; d < headDim; d++) {
            oSeg.setAtIndex(ValueLayout.JAVA_FLOAT, (oOff / 4) + d, acc[d] / l);
        }
    }

    private static float dotProduct(MemorySegment q, long qOff, MemorySegment k, long kOff, long dim) {
        FloatVector sum = FloatVector.zero(SPECIES);
        int j = 0;
        for (; j < SPECIES.loopBound(dim); j += SPECIES.length()) {
            FloatVector vq = FloatVector.fromMemorySegment(SPECIES, q, qOff + (long)j * 4, java.nio.ByteOrder.nativeOrder());
            FloatVector vk = FloatVector.fromMemorySegment(SPECIES, k, kOff + (long)j * 4, java.nio.ByteOrder.nativeOrder());
            sum = vq.fma(vk, sum);
        }
        float res = sum.reduceLanes(VectorOperators.ADD);
        for (; j < dim; j++) {
            res += q.getAtIndex(ValueLayout.JAVA_FLOAT, (qOff / 4) + j) * 
                   k.getAtIndex(ValueLayout.JAVA_FLOAT, (kOff / 4) + j);
        }
        return res;
    }

    private static void updateAccumulator(float[] acc, MemorySegment vSeg, long vOff, float exp_prev, float exp_curr, long dim) {
        FloatVector vExpPrev = FloatVector.broadcast(SPECIES, exp_prev);
        FloatVector vExpCurr = FloatVector.broadcast(SPECIES, exp_curr);
        int j = 0;
        for (; j < SPECIES.loopBound(dim); j += SPECIES.length()) {
            FloatVector vacc = FloatVector.fromArray(SPECIES, acc, j);
            FloatVector vv = FloatVector.fromMemorySegment(SPECIES, vSeg, vOff + (long)j * 4, java.nio.ByteOrder.nativeOrder());
            FloatVector res = vacc.mul(vExpPrev).fma(vExpCurr, vv);
            res.intoArray(acc, j);
        }
        for (; j < dim; j++) {
            acc[j] = acc[j] * exp_prev + vSeg.getAtIndex(ValueLayout.JAVA_FLOAT, (vOff / 4) + j) * exp_curr;
        }
    }
}
