package tech.kayys.gollek.inference.nativeimpl;

import java.lang.foreign.MemorySegment;
import tech.kayys.gollek.spi.tensor.ComputeKernel;
import tech.kayys.gollek.spi.tensor.ComputeKernelRegistry;

/**
 * Small adapter that chooses the best available ComputeKernel and delegates flash attention calls.
 */
public final class FlashAttentionAdapter {

    private static final ComputeKernel KERNEL = ComputeKernelRegistry.get().getBestAvailable();

    private FlashAttentionAdapter() {}

    public static void flashAttention(MemorySegment out, MemorySegment q, MemorySegment k, MemorySegment v,
                                      int seqLen, int numHeads, int headDim) {
        KERNEL.flashAttention(out, q, k, v, seqLen, numHeads, headDim);
    }
}
