package tech.kayys.gollek.kernel.paged;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import org.jboss.logging.Logger;

/**
 * Pure-Java CPU fallback for PagedAttention.
 * <p>
 * This implementation is functionally equivalent to the CUDA kernel but
 * runs on CPU. It is significantly slower (no parallelism, no GPU) but
 * enables development and testing without a GPU.
 * <p>
 * <b>Memory layout assumptions:</b>
 * <ul>
 *   <li>All tensors use float32 (4 bytes per element)</li>
 *   <li>Query: [numSeqs, numHeads, headDim]</li>
 *   <li>K/V Cache: [totalBlocks, numHeads, blockSize, headDim]</li>
 *   <li>Block tables: [numSeqs, maxBlocksPerSeq] (int32)</li>
 *   <li>Context lens: [numSeqs] (int32)</li>
 * </ul>
 */
public final class PagedAttentionCpuFallback {

    private static final Logger LOG = Logger.getLogger(PagedAttentionCpuFallback.class);

    private PagedAttentionCpuFallback() {
    }

    /**
     * Execute PagedAttention on CPU.
     * Signature matches the CUDA kernel for drop-in replacement.
     *
     * @return 0 on success
     */
    public static int execute(
            MemorySegment output,
            MemorySegment query,
            MemorySegment keyCache,
            MemorySegment valueCache,
            MemorySegment blockTables,
            MemorySegment contextLens,
            int numSeqs,
            int numHeads,
            int headDim,
            int blockSize,
            int maxBlocksPerSeq,
            float scale
    ) {
        for (int seq = 0; seq < numSeqs; seq++) {
            int contextLen = contextLens.getAtIndex(ValueLayout.JAVA_INT, seq);
            int numBlocks = (contextLen + blockSize - 1) / blockSize;

            for (int head = 0; head < numHeads; head++) {
                // Phase 1: Compute attention scores (Q · K^T)
                float[] scores = new float[contextLen];
                float maxScore = Float.NEGATIVE_INFINITY;

                for (int blockIdx = 0; blockIdx < numBlocks; blockIdx++) {
                    int physicalBlock = blockTables.getAtIndex(
                            ValueLayout.JAVA_INT,
                            (long) seq * maxBlocksPerSeq + blockIdx
                    );
                    int tokensInBlock = Math.min(blockSize, contextLen - blockIdx * blockSize);

                    for (int tok = 0; tok < tokensInBlock; tok++) {
                        int absPos = blockIdx * blockSize + tok;
                        float dot = 0.0f;

                        for (int d = 0; d < headDim; d++) {
                            // Query layout: [seq, head, d]
                            long qIdx = ((long) seq * numHeads + head) * headDim + d;
                            float qVal = query.getAtIndex(ValueLayout.JAVA_FLOAT, qIdx) * scale;

                            // K cache layout: [block, head, tok, d]
                            long kIdx = (((long) physicalBlock * numHeads + head) * blockSize + tok) * headDim + d;
                            float kVal = keyCache.getAtIndex(ValueLayout.JAVA_FLOAT, kIdx);

                            dot += qVal * kVal;
                        }

                        scores[absPos] = dot;
                        maxScore = Math.max(maxScore, dot);
                    }
                }

                // Phase 2: Softmax
                float sumExp = 0.0f;
                for (int i = 0; i < contextLen; i++) {
                    scores[i] = (float) Math.exp(scores[i] - maxScore);
                    sumExp += scores[i];
                }
                for (int i = 0; i < contextLen; i++) {
                    scores[i] /= sumExp;
                }

                // Phase 3: Weighted sum of V
                for (int d = 0; d < headDim; d++) {
                    float acc = 0.0f;

                    for (int blockIdx = 0; blockIdx < numBlocks; blockIdx++) {
                        int physicalBlock = blockTables.getAtIndex(
                                ValueLayout.JAVA_INT,
                                (long) seq * maxBlocksPerSeq + blockIdx
                        );
                        int tokensInBlock = Math.min(blockSize, contextLen - blockIdx * blockSize);

                        for (int tok = 0; tok < tokensInBlock; tok++) {
                            int absPos = blockIdx * blockSize + tok;

                            // V cache layout: [block, head, tok, d]
                            long vIdx = (((long) physicalBlock * numHeads + head) * blockSize + tok) * headDim + d;
                            acc += scores[absPos] * valueCache.getAtIndex(ValueLayout.JAVA_FLOAT, vIdx);
                        }
                    }

                    // Output layout: [seq, head, d]
                    long outIdx = ((long) seq * numHeads + head) * headDim + d;
                    output.setAtIndex(ValueLayout.JAVA_FLOAT, outIdx, acc);
                }
            }
        }

        return 0;
    }
}
