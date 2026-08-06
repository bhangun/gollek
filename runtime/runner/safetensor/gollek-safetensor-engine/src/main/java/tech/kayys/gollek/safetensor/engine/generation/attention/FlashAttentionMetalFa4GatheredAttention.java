/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import tech.kayys.alkhawarizm.metal.binding.MetalFlashAttentionBinding;
import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.DirectInferenceProfiler;
import tech.kayys.gollek.safetensor.engine.generation.kv.BlockManager;
import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.function.Supplier;

final class FlashAttentionMetalFa4GatheredAttention {
    private final Supplier<MetalFlashAttentionBinding> metalFa4;
    private final FlashAttentionPrecisionPolicy precisionPolicy;

    FlashAttentionMetalFa4GatheredAttention(Supplier<MetalFlashAttentionBinding> metalFa4) {
        this(metalFa4, FlashAttentionPrecisionOptions.defaults());
    }

    FlashAttentionMetalFa4GatheredAttention(Supplier<MetalFlashAttentionBinding> metalFa4,
            FlashAttentionPrecisionOptions precisionOptions) {
        this.metalFa4 = metalFa4;
        this.precisionPolicy = FlashAttentionPrecisionPolicy.from(
                precisionOptions == null ? FlashAttentionPrecisionOptions.defaults() : precisionOptions);
    }

    AccelTensor tryCompute(AccelTensor q, KVCacheManager.KVCacheSession kvSession, BlockManager blockManager,
            int kvLayerIdx, int totalTokens, int numHeads, int numKVHeads, int headDim, float scale, boolean causal,
            float softCap, long batch, long seqLen, Arena arena, String successPath,
            MemorySegment attentionContextBuffer) {
        MetalFlashAttentionBinding fa4 = metalFa4();
        if (fa4 == null || !fa4.isNativeAvailable()) {
            return null;
        }
        AccelTensor out = FlashAttentionContextOutputBuffer.viewOrAllocate(attentionContextBuffer, q);
        boolean success = false;
        try {
            long gatherBytes = (long) totalTokens * numKVHeads * headDim * Float.BYTES;
            MemorySegment kGathered = arena.allocate(gatherBytes, 64);
            MemorySegment vGathered = arena.allocate(gatherBytes, 64);
            PagedKvCacheIO.gather(blockManager, kvSession, kvLayerIdx, totalTokens, numKVHeads, headDim,
                    kGathered, vGathered);

            boolean useBf16 = precisionPolicy.useBf16Attention(fa4);
            int result = fa4.fa4Attention(
                    out.dataPtr(), q.dataPtr(), kGathered, vGathered,
                    (int) batch, (int) seqLen, totalTokens, numHeads, numKVHeads, headDim,
                    scale, causal, useBf16, softCap);
            if (result == 0) {
                success = true;
                DirectInferenceProfiler.recordAttentionPath(successPath);
                return out;
            }
            return null;
        } finally {
            if (!success && !out.isClosed()) {
                out.close();
            }
        }
    }

    private MetalFlashAttentionBinding metalFa4() {
        return metalFa4.get();
    }
}
