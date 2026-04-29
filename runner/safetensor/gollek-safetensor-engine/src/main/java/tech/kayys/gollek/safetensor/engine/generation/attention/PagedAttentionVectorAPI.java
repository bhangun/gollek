package tech.kayys.gollek.safetensor.engine.generation.attention;

import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.kv.BlockManager;
import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.List;

/**
 * Paged Attention-2 implementation using Java scalar loops (fallback).
 * Traverses a block-mapped KV cache for memory-efficient inference.
 * Matches the layout: [numHeads, tokensPerBlock, headDim]
 */
public class PagedAttentionVectorAPI {

    /**
     * Compute attention using the Paged Attention algorithm.
     */
    public static AccelTensor compute(
            AccelTensor q,
            KVCacheManager.KVCacheSession kvSession,
            int layerIdx,
            int startPos,
            int numQHeads,
            int headDim,
            float scale,
            boolean causal,
            float softCap) {

        long batch = q.size(0);
        long seqLenQ = q.size(1);
        int totalTokens = startPos + (int) seqLenQ;
        int numKVHeads = kvSession.numKVHeads();
        int tokensPerBlock = kvSession.tokensPerBlock();
        List<Integer> blockTable = kvSession.getBlockIndices(layerIdx);
        BlockManager blockManager = kvSession.blockManager();

        AccelTensor out = AccelTensor.zeros(q.shape());

        for (int b = 0; b < batch; b++) {
            for (int h = 0; h < numQHeads; h++) {
                for (int i = 0; i < seqLenQ; i++) {
                    computePagedHeadQuery(b, h, i, q, blockTable, blockManager, out, scale, causal, tokensPerBlock,
                            totalTokens, numKVHeads, headDim, softCap);
                }
            }
        }
        return out;
    }

    private static void computePagedHeadQuery(int b, int h, int i,
            AccelTensor q, List<Integer> blockTable, BlockManager blockManager, AccelTensor out,
            float scale, boolean causal, int tokensPerBlock, int totalTokens, int numKVHeads, int headDim, float softCap) {

        MemorySegment qSeg = q.dataSegment();
        MemorySegment oSeg = out.dataSegment();

        long qStride0 = q.stride()[0];
        long qStride1 = q.stride()[1];
        long qStride2 = q.stride()[2];

        // Correct offset accounting for tensor strides
        long qOff = ((long) b * qStride0 + (long) i * qStride1 + (long) h * qStride2) * 4;

        float m = Float.NEGATIVE_INFINITY;
        float l = 0.0f;
        float[] acc = new float[headDim];

        int groupSize = (int) (q.size(2) / numKVHeads); // Wait, q shape is [B, T, H, D]
        int kvH = h / (int)(q.size(2) / numKVHeads); 
        // Better:
        int qHeads = (int)q.size(2);
        int gqaGroup = qHeads / numKVHeads;
        int kvHeadIdx = h / gqaGroup;

        long headStride = blockManager.getHeadStride();
        long tokenStride = blockManager.getTokenStride();

        // Softmax Online Loop over blocks
        int currentTokCount = 0;
        for (int blockIdx : blockTable) {
            MemorySegment kBlock = blockManager.getKBlock(blockIdx);
            MemorySegment vBlock = blockManager.getVBlock(blockIdx);
            int tokensInThisBlock = Math.min(tokensPerBlock, totalTokens - currentTokCount);

            for (int tok = 0; tok < tokensInThisBlock; tok++) {
                int absPos = currentTokCount + tok;
                int seqLenQ = (int)q.size(1);
                int startPos = totalTokens - seqLenQ;
                if (causal && absPos > startPos + i)
                    break;

                // 1. Compute Dot Product Q_i @ K_j
                long kvOff = ((long) kvHeadIdx * headStride + (long) tok * tokenStride) * 4;
                float score = dotProduct(qSeg, qOff, kBlock, kvOff, headDim) * scale;
                
                if (softCap > 0.0f) {
                    score = (float) (Math.tanh(score / softCap) * softCap);
                }

                // 2. Online Softmax Update
                float m_prev = m;
                m = Math.max(m, score);
                float exp_prev = (float) Math.exp(m_prev - m);
                float exp_curr = (float) Math.exp(score - m);
                l = l * exp_prev + exp_curr;

                // 3. Update Output Accumulator
                updateAccumulator(acc, vBlock, kvOff, exp_prev, exp_curr, headDim);
            }

            currentTokCount += tokensInThisBlock;
            if (currentTokCount >= totalTokens)
                break;
        }

        // 4. Final Normalization
        long oStride0 = out.stride()[0];
        long oStride1 = out.stride()[1];
        long oStride2 = out.stride()[2];
        long oOff = ((long) b * oStride0 + (long) i * oStride1 + (long) h * oStride2) * 4;

        for (int d = 0; d < headDim; d++) {
            oSeg.setAtIndex(ValueLayout.JAVA_FLOAT, (oOff / 4) + d, acc[d] / (l + 1e-9f));
        }
    }

    private static float dotProduct(MemorySegment q, long qOff, MemorySegment k, long kOff, long dim) {
        float res = 0;
        for (int j = 0; j < dim; j++) {
            res += q.getAtIndex(ValueLayout.JAVA_FLOAT, (qOff / 4) + j) *
                    k.getAtIndex(ValueLayout.JAVA_FLOAT, (kOff / 4) + j);
        }
        return res;
    }

    private static void updateAccumulator(float[] acc, MemorySegment vSeg, long vOff, float exp_prev, float exp_curr,
            long dim) {
        for (int j = 0; j < dim; j++) {
            acc[j] = acc[j] * exp_prev + vSeg.getAtIndex(ValueLayout.JAVA_FLOAT, (vOff / 4) + j) * exp_curr;
        }
    }
}
