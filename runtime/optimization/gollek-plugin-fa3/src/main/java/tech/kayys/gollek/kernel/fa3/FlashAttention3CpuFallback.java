package tech.kayys.gollek.kernel.fa3;

import java.lang.foreign.MemorySegment;

import org.jboss.logging.Logger;

/**
 * Cpu fallback for FlashAttention-3 when Hopper GPUs or native libraries are unavailable.
 * <p>
 * This is a standard, memory-bandwidth bound attention implementation
 * computing Q*K^T / sqrt(d) -> Softmax -> * V in standard Java.
 * Not suitable for production serving, primarily for testing and fallback.
 */
public class FlashAttention3CpuFallback {

    private static final Logger LOG = Logger.getLogger(FlashAttention3CpuFallback.class);

    public static int execute(
            MemorySegment output,
            MemorySegment query,
            MemorySegment key,
            MemorySegment value,
            int batchSize,
            int seqLen,
            int numHeads,
            int numHeadsK,
            int headDim,
            float softmaxScale,
            boolean isCausal,
            boolean useFp8
    ) {
        LOG.debug("Executing standard attention CPU fallback (FA3 Native unavailable)");

        if (useFp8) {
            LOG.warn("FP8 is not supported in the CPU fallback. Falling back to FP32 calculation assuming inputs are FP32.");
        }

        java.lang.foreign.ValueLayout.OfFloat FLOAT = java.lang.foreign.ValueLayout.JAVA_FLOAT;
        int groupSize = numHeads / numHeadsK;

        for (int b = 0; b < batchSize; b++) {
            for (int h = 0; h < numHeads; h++) {
                int hk = h / groupSize; // Grouped Query Attention mapping
                
                for (int q = 0; q < seqLen; q++) {
                    float[] scores = new float[seqLen];
                    float maxScore = Float.NEGATIVE_INFINITY;
                    
                    // 1. Q * K^T
                    for (int k = 0; k < seqLen; k++) {
                        if (isCausal && k > q) {
                            scores[k] = Float.NEGATIVE_INFINITY;
                            continue;
                        }
                        
                        float score = 0.0f;
                        for (int d = 0; d < headDim; d++) {
                            long qOffset = ((long) b * seqLen * numHeads * headDim 
                                          + (long) q * numHeads * headDim 
                                          + (long) h * headDim 
                                          + d) * 4L;
                            long kOffset = ((long) b * seqLen * numHeadsK * headDim 
                                          + (long) k * numHeadsK * headDim 
                                          + (long) hk * headDim 
                                          + d) * 4L;
                                          
                            score += query.get(FLOAT, qOffset) * key.get(FLOAT, kOffset);
                        }
                        score *= softmaxScale;
                        scores[k] = score;
                        if (score > maxScore) {
                            maxScore = score;
                        }
                    }
                    
                    // 2. Softmax
                    float sumExp = 0.0f;
                    int maxK = isCausal ? q : seqLen - 1;
                    for (int k = 0; k <= maxK; k++) {
                        scores[k] = (float) Math.exp(scores[k] - maxScore);
                        sumExp += scores[k];
                    }
                    for (int k = 0; k <= maxK; k++) {
                        scores[k] /= sumExp;
                    }
                    
                    // 3. Scores * V
                    for (int d = 0; d < headDim; d++) {
                        float outVal = 0.0f;
                        for (int k = 0; k <= maxK; k++) {
                            long vOffset = ((long) b * seqLen * numHeadsK * headDim 
                                          + (long) k * numHeadsK * headDim 
                                          + (long) hk * headDim 
                                          + d) * 4L;
                            outVal += scores[k] * value.get(FLOAT, vOffset);
                        }
                        
                        long oOffset = ((long) b * seqLen * numHeads * headDim 
                                      + (long) q * numHeads * headDim 
                                      + (long) h * headDim 
                                      + d) * 4L;
                        output.set(FLOAT, oOffset, outVal);
                    }
                }
            }
        }
        
        return 0; // Success code
    }
}
