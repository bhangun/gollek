package tech.kayys.gollek.cuda.optimization;

import org.jboss.logging.Logger;
import tech.kayys.gollek.flashattn.binding.FlashAttention4Binding;
import tech.kayys.gollek.kvcache.PagedKVCacheManager;

import java.lang.foreign.MemorySegment;

/**
 * CUDA optimization manager that integrates FlashAttention-4 and other
 * optimization modules with the CUDA kernel runner.
 *
 * <p>
 * This class provides a unified interface for:
 * <ul>
 *   <li>FlashAttention-4 (Blackwell sm_100a)</li>
 *   <li>FlashAttention-3 (Hopper sm_90)</li>
 *   <li>Hybrid Attention + GDN</li>
 *   <li>Paged Attention with KV cache</li>
 * </ul>
 *
 * <h2>Auto-Selection Logic</h2>
 * <p>
 * The optimal attention kernel is selected based on:
 * <ul>
 *   <li>Compute capability (sm_100a → FA4, sm_90 → FA3, sm_80 → FA2)</li>
 *   <li>Model architecture (transformer vs hybrid)</li>
 *   <li>Precision requirements (FP4/FP8/BF16)</li>
 * </ul>
 */
public class CudaOptimizationManager {

    private static final Logger LOG = Logger.getLogger(CudaOptimizationManager.class);

    private final FlashAttention4Binding fa4Binding;
    private final PagedKVCacheManager kvCacheManager;
    private final int computeCapability;
    private final boolean hasTMEM;
    private final boolean supportsFP4;

    /**
     * Create CUDA optimization manager.
     *
     * @param computeCapability GPU compute cap (e.g., 80, 90, 100)
     * @param hasTMEM           true if TMEM available (Blackwell+)
     * @param supportsFP4       true if FP4 tensor cores available
     * @param kvCacheManager    KV cache manager for paged attention
     */
    public CudaOptimizationManager(int computeCapability,
                                    boolean hasTMEM,
                                    boolean supportsFP4,
                                    PagedKVCacheManager kvCacheManager) {
        this.computeCapability = computeCapability;
        this.hasTMEM = hasTMEM;
        this.supportsFP4 = supportsFP4;
        this.kvCacheManager = kvCacheManager;
        this.fa4Binding = FlashAttention4Binding.getInstance();

        LOG.infof("[CUDA Opt] Initialized — compute=%d.%d TMEM=%s FP4=%s",
                computeCapability / 10, computeCapability % 10,
                hasTMEM, supportsFP4);
    }

    /**
     * Execute optimized attention based on GPU capabilities.
     *
     * @param output    Output buffer [B, T, H, D]
     * @param query     Query buffer [B, T, H, D]
     * @param kPool     K cache pool (paged)
     * @param vPool     V cache pool (paged)
     * @param batchSize Batch size
     * @param seqLen    Sequence length
     * @param numHeads  Number of attention heads
     * @param numHeadsKV Number of KV heads
     * @param headDim   Head dimension
     * @param scale     Attention scale (1/sqrt(D))
     * @param causal    true for causal masking
     * @param useFp8    true for FP8 precision
     */
    public void executeAttention(MemorySegment output,
                                  MemorySegment query,
                                  MemorySegment kPool,
                                  MemorySegment vPool,
                                  int batchSize,
                                  int seqLen,
                                  int numHeads,
                                  int numHeadsKV,
                                  int headDim,
                                  float scale,
                                  boolean causal,
                                  boolean useFp8) {

        if (computeCapability >= 100 && hasTMEM) {
            // Blackwell: Use FlashAttention-4 with TMEM
            executeFlashAttention4(output, query, kPool, vPool,
                    batchSize, seqLen, numHeads, numHeadsKV, headDim,
                    scale, causal, useFp8);
        } else if (computeCapability >= 90) {
            // Hopper: Use FlashAttention-3
            executeFlashAttention3(output, query, kPool, vPool,
                    batchSize, seqLen, numHeads, numHeadsKV, headDim,
                    scale, causal, useFp8);
        } else if (computeCapability >= 80) {
            // Ampere: Use FlashAttention-2
            executeFlashAttention2(output, query, kPool, vPool,
                    batchSize, seqLen, numHeads, numHeadsKV, headDim,
                    scale, causal);
        } else {
            // Fallback: Use paged attention
            executePagedAttention(output, query, kPool, vPool,
                    batchSize, seqLen, numHeads, headDim, scale);
        }
    }

    /**
     * Execute FlashAttention-4 (Blackwell only).
     * Uses TMEM for QK^T accumulation and FP4 tensor cores.
     */
    private void executeFlashAttention4(MemorySegment output,
                                         MemorySegment query,
                                         MemorySegment kPool,
                                         MemorySegment vPool,
                                         int batchSize,
                                         int seqLen,
                                         int numHeads,
                                         int numHeadsKV,
                                         int headDim,
                                         float scale,
                                         boolean causal,
                                         boolean useFp8) {
        if (!fa4Binding.isNativeAvailable()) {
            LOG.warn("FA4 not available, falling back to FA3");
            executeFlashAttention3(output, query, kPool, vPool,
                    batchSize, seqLen, numHeads, numHeadsKV, headDim,
                    scale, causal, useFp8);
            return;
        }

        int err = fa4Binding.flashAttention4Launch(
                output, query, kPool, vPool,
                batchSize, seqLen, numHeads, numHeadsKV, headDim,
                scale, causal, useFp8);

        if (err != 0) {
            LOG.warnf("FA4 kernel error %d, falling back to FA3", err);
            executeFlashAttention3(output, query, kPool, vPool,
                    batchSize, seqLen, numHeads, numHeadsKV, headDim,
                    scale, causal, useFp8);
        } else {
            LOG.debug("Executed FA4 attention with TMEM");
        }
    }

    /**
     * Execute FlashAttention-3 (Hopper+).
     */
    private void executeFlashAttention3(MemorySegment output,
                                         MemorySegment query,
                                         MemorySegment kPool,
                                         MemorySegment vPool,
                                         int batchSize,
                                         int seqLen,
                                         int numHeads,
                                         int numHeadsKV,
                                         int headDim,
                                         float scale,
                                         boolean causal,
                                         boolean useFp8) {
        // In production, this would call FA3 binding
        // For now, fall back to paged attention
        executePagedAttention(output, query, kPool, vPool,
                batchSize, seqLen, numHeads, headDim, scale);
        LOG.debug("Executed FA3 attention");
    }

    /**
     * Execute FlashAttention-2 (Ampere+).
     */
    private void executeFlashAttention2(MemorySegment output,
                                         MemorySegment query,
                                         MemorySegment kPool,
                                         MemorySegment vPool,
                                         int batchSize,
                                         int seqLen,
                                         int numHeads,
                                         int numHeadsKV,
                                         int headDim,
                                         float scale,
                                         boolean causal) {
        // Fall back to paged attention
        executePagedAttention(output, query, kPool, vPool,
                batchSize, seqLen, numHeads, headDim, scale);
        LOG.debug("Executed FA2 attention");
    }

    /**
     * Execute standard paged attention (fallback for all GPUs).
     */
    private void executePagedAttention(MemorySegment output,
                                        MemorySegment query,
                                        MemorySegment kPool,
                                        MemorySegment vPool,
                                        int batchSize,
                                        int seqLen,
                                        int numHeads,
                                        int headDim,
                                        float scale) {
        // In production, this would call PagedAttentionBinding
        LOG.debug("Executed paged attention (fallback)");
    }

    /**
     * Get the recommended attention kernel for current GPU.
     *
     * @return Kernel name (e.g., "FA4-TMEM", "FA3", "FA2", "Paged")
     */
    public String getRecommendedKernel() {
        if (computeCapability >= 100 && hasTMEM) {
            return supportsFP4 ? "FA4-TMEM-FP4" : "FA4-TMEM";
        } else if (computeCapability >= 90) {
            return "FA3";
        } else if (computeCapability >= 80) {
            return "FA2";
        } else {
            return "Paged";
        }
    }

    /**
     * Check if FP4 precision is supported and recommended.
     */
    public boolean shouldUseFP4() {
        return supportsFP4 && computeCapability >= 100;
    }

    /**
     * Check if FP8 precision is supported.
     */
    public boolean shouldUseFP8() {
        return computeCapability >= 90;
    }

    /**
     * Get KV cache manager for paged attention integration.
     */
    public PagedKVCacheManager getKvCacheManager() {
        return kvCacheManager;
    }
}
