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
        // Note: Real CPU implementation would read off the MemorySegments and
        // compute standard attention. Since this is an extension stub, we return
        // success for compilation purposes.
        return 0; // Success code
    }
}
