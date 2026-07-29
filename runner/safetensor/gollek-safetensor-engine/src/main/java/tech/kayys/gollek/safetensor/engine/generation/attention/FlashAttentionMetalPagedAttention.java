/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import tech.kayys.alkhawarizm.metal.binding.MetalBinding;
import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.kv.BlockManager;
import tech.kayys.gollek.safetensor.engine.generation.kv.ForwardWorkspace;
import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;
import tech.kayys.gollek.spi.model.ModelConfig;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.List;
import java.util.function.Supplier;

final class FlashAttentionMetalPagedAttention {
    private final Supplier<MetalBinding> metalBinding;
    private final Supplier<FlashAttentionRoutingPolicy> routingPolicy;

    FlashAttentionMetalPagedAttention(Supplier<MetalBinding> metalBinding,
            Supplier<FlashAttentionRoutingPolicy> routingPolicy) {
        this.metalBinding = metalBinding;
        this.routingPolicy = routingPolicy;
    }

    AccelTensor tryCompute(AccelTensor q, KVCacheManager.KVCacheSession kvSession,
            BlockManager blockManager, List<Integer> blocks, int kvLayerIdx, int startPos, int numHeads,
            int numKVHeads, int headDim, float scale, boolean causal, float softCap, ModelConfig config, int layerIdx,
            int totalTokens, long batch, long seqLen, boolean slidingLayer, Arena arena,
            FlashAttentionModelPolicy modelPolicy, MemorySegment attentionContextBuffer) {
        MetalBinding binding = metalBinding();
        if (!routingPolicy().allowPagedMetalAttentionBridge(modelPolicy, (int) seqLen, totalTokens)
                || binding == null
                || !binding.isRuntimeActive()
                || blocks == null
                || blocks.isEmpty()) {
            return null;
        }
        if (kvSession.isQuantized() && arena == null) {
            return null;
        }
        AccelTensor out = FlashAttentionContextOutputBuffer.viewOrAllocate(attentionContextBuffer, q);
        try {
            int maxBlocks = blocks.size();
            int batchInt = Math.toIntExact(batch);
            MemorySegment kPool = kvSession.getRawKPool();
            MemorySegment vPool = kvSession.getRawVPool();
            ForwardWorkspace ws = kvSession.getWorkspace();
            ws.ensureAttentionMetadataCapacity(batchInt * maxBlocks, batchInt);
            MemorySegment blockTableSegment = ws.getAttentionBlockTableSeg();
            MemorySegment contextLensSegment = ws.getAttentionContextLensSeg();
            if (kvSession.isQuantized()) {
                PagedKvCacheMaterializer.MaterializedKvPools materialized = PagedKvCacheMaterializer
                        .materializeBlocksForMetal(
                                blockManager, kvSession, blocks, numKVHeads, headDim, arena);
                kPool = materialized.kPool();
                vPool = materialized.vPool();
                for (int b = 0; b < batchInt; b++) {
                    int base = b * maxBlocks;
                    for (int i = 0; i < maxBlocks; i++) {
                        blockTableSegment.setAtIndex(ValueLayout.JAVA_INT, base + i, i);
                    }
                }
            } else {
                for (int b = 0; b < batchInt; b++) {
                    int base = b * maxBlocks;
                    for (int i = 0; i < maxBlocks; i++) {
                        blockTableSegment.setAtIndex(ValueLayout.JAVA_INT, base + i, blocks.get(i));
                    }
                }
            }
            for (int b = 0; b < batchInt; b++) {
                contextLensSegment.setAtIndex(ValueLayout.JAVA_INT, b, totalTokens);
            }

            int result = slidingLayer
                    ? (numKVHeads == numHeads
                            ? binding.attentionWindowed(
                                    out.dataPtr(), q.dataPtr(), kPool, vPool,
                                    blockTableSegment, contextLensSegment,
                                    (int) batch, (int) seqLen, numHeads, numKVHeads, headDim,
                                    kvSession.tokensPerBlock(), maxBlocks,
                                    scale, causal ? 1 : 0, startPos, config.getSlidingWindowSize(), softCap)
                            : binding.attentionGqaWindowed(
                                    out.dataPtr(), q.dataPtr(), kPool, vPool,
                                    blockTableSegment, contextLensSegment,
                                    (int) batch, (int) seqLen, numHeads, numKVHeads, headDim,
                                    kvSession.tokensPerBlock(), maxBlocks,
                                    scale, causal ? 1 : 0, startPos, config.getSlidingWindowSize(), softCap))
                    : (numKVHeads == numHeads
                            ? binding.attention(
                                    out.dataPtr(), q.dataPtr(), kPool, vPool,
                                    blockTableSegment, contextLensSegment,
                                    (int) batch, (int) seqLen, numHeads, headDim,
                                    kvSession.tokensPerBlock(), maxBlocks,
                                    scale, causal ? 1 : 0, softCap)
                            : binding.attentionGqa(
                                    out.dataPtr(), q.dataPtr(), kPool, vPool,
                                    blockTableSegment, contextLensSegment,
                                    (int) batch, (int) seqLen, numHeads, numKVHeads, headDim,
                                    kvSession.tokensPerBlock(), maxBlocks,
                                    scale, causal ? 1 : 0, softCap));
            if (result == 0) {
                return out;
            }
        } catch (RuntimeException e) {
            // Fall through to FA4/Java attention fallback below.
        }
        out.close();
        return null;
    }

    private MetalBinding metalBinding() {
        return metalBinding.get();
    }

    private FlashAttentionRoutingPolicy routingPolicy() {
        return routingPolicy.get();
    }
}
