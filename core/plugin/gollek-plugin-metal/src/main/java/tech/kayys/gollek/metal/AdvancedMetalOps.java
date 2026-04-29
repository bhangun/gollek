package tech.kayys.gollek.metal;

import java.lang.foreign.MemorySegment;

/**
 * Extensible interface exposing advanced, fused operations natively available in Metal MPS.
 * 
 * <p>High-level orchestration runners (like Safetensor or GGUF generation engines)
 * can query if their active {@code ComputeBackend} implements this interface. 
 * If so, they can bypass sequential generic array iterations and leverage these directly
 * for drastic latency reductions.
 */
public interface AdvancedMetalOps {

    /**
     * Executes fused RMS Normalization: out = x / rms(x) * weight
     */
    void rmsNorm(float[] out, float[] x, float[] weight, int n, float eps, boolean addOne);
    void rmsNorm(MemorySegment out, MemorySegment x, MemorySegment weight, int n, float eps, boolean addOne);

    /**
     * Executes SiLU-gated Feed Forward Network (often found in Llama, Qwen):
     * out = silu(gate) * up
     */
    void siluFfn(float[] out, float[] gate, float[] up, int n);
    void siluFfn(MemorySegment out, MemorySegment gate, MemorySegment up, int n);

    /**
     * Executes full scaled-dot product attention natively on the MPS tensor cores,
     * reading from off-heap unified memory slabs (Paged Attention layout).
     */
    void pagedAttention(
            MemorySegment out, 
            MemorySegment q,
            MemorySegment kCache, 
            MemorySegment vCache,
            MemorySegment blockTable, 
            MemorySegment contextLens,
            int b, int t, int h, int d,
            int blockSize, int maxBlocks,
            float scale, boolean isCausal, float softCap);

}
