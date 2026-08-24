/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import tech.kayys.alkhawarizm.metal.binding.MetalBinding;
import tech.kayys.alkhawarizm.metal.binding.MetalFlashAttentionBinding;
import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.DirectInferenceProfiler;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.function.BooleanSupplier;
import java.util.function.Supplier;

final class FlashAttentionMetalDenseSharedAttention {
    private final BooleanSupplier canUseMetal;
    private final Supplier<MetalBinding> metalBinding;
    private final Supplier<MetalFlashAttentionBinding> metalFa4;
    private final Supplier<FlashAttentionRoutingPolicy> routingPolicy;
    private final FlashAttentionPrecisionPolicy precisionPolicy;

    FlashAttentionMetalDenseSharedAttention(BooleanSupplier canUseMetal, Supplier<MetalBinding> metalBinding,
            Supplier<MetalFlashAttentionBinding> metalFa4,
            Supplier<FlashAttentionRoutingPolicy> routingPolicy) {
        this(canUseMetal, metalBinding, metalFa4, routingPolicy, FlashAttentionPrecisionOptions.defaults());
    }

    FlashAttentionMetalDenseSharedAttention(BooleanSupplier canUseMetal, Supplier<MetalBinding> metalBinding,
            Supplier<MetalFlashAttentionBinding> metalFa4,
            Supplier<FlashAttentionRoutingPolicy> routingPolicy,
            FlashAttentionPrecisionOptions precisionOptions) {
        this.canUseMetal = canUseMetal;
        this.metalBinding = metalBinding;
        this.metalFa4 = metalFa4;
        this.routingPolicy = routingPolicy;
        this.precisionPolicy = FlashAttentionPrecisionPolicy.from(
                precisionOptions == null ? FlashAttentionPrecisionOptions.defaults() : precisionOptions);
    }

    AccelTensor compute(AccelTensor q, AccelTensor k, AccelTensor v,
            SharedKvState sharedKvState, ModelConfig config, FlashAttentionModelPolicy modelPolicy,
            int layerIdx, int startPos, int numQHeads, int numKVHeads, int headDim,
            float scale, boolean causal, float softCap, MemorySegment attentionContextBuffer) {
        if (!canUseMetal.getAsBoolean()) {
            return null;
        }
        MetalBinding binding = metalBinding();
        if (binding == null || !binding.isRuntimeActive()) {
            return null;
        }
        long batch = q.size(0);
        long seqLenQ = q.size(1);
        int totalTokens = Math.toIntExact(k.size(1));
        if (batch <= 0 || seqLenQ <= 0 || totalTokens <= 0) {
            return null;
        }

        boolean slidingLayer = config != null && config.isSlidingAttentionLayer(layerIdx) && config.hasSlidingWindow();
        if (slidingLayer && !binding.isWindowedAttentionAvailable()) {
            return null;
        }

        FlashAttentionRoutingPolicy routing = routingPolicy();
        boolean usePackedSharedDecode = routing.shouldUsePackedSharedDecodeAttention(config, modelPolicy, seqLenQ,
                sharedKvState);
        boolean useFa4 = false;
        if (!usePackedSharedDecode && routing.canUseFa4Attention(softCap)) {
            if (!slidingLayer) {
                useFa4 = true;
            } else {
                useFa4 = routing.canUseRestrictedSlidingPrefillFa4Attention(config, layerIdx, Math.toIntExact(seqLenQ),
                        startPos, softCap);
            }
        }

        if (!useFa4 && !usePackedSharedDecode && !routing.allowLegacyMetalAttentionBridge(modelPolicy)) {
            return null;
        }
        AccelTensor qContiguous = q.contiguous();
        AccelTensor kContiguous = null;
        AccelTensor vContiguous = null;
        AccelTensor out = null;
        try (Arena arena = Arena.ofConfined()) {
            out = FlashAttentionContextOutputBuffer.viewOrAllocate(attentionContextBuffer, q);

            if (useFa4) {
                MetalFlashAttentionBinding fa4 = metalFa4();
                if (fa4 == null || !fa4.isNativeAvailable()) {
                    out.close();
                    return null;
                }
                kContiguous = k.contiguous();
                vContiguous = v.contiguous();
                boolean useBf16 = precisionPolicy.useBf16Attention(fa4);
                int result = fa4.fa4Attention(
                        out.dataPtr(), qContiguous.dataPtr(), kContiguous.dataPtr(), vContiguous.dataPtr(),
                        Math.toIntExact(batch), Math.toIntExact(seqLenQ), totalTokens, numQHeads, numKVHeads, headDim,
                        scale, causal, useBf16, softCap);
                if (result == 0) {
                    DirectInferenceProfiler.recordAttentionPath("dense_shared_fa4");
                    return out;
                }
                out.close();
            } else {
                int blockSize = Math.max(1, sharedKvState != null
                        ? sharedKvState.packedCapacityTokens()
                        : totalTokens);
                int maxBlocks = 1;
                MemorySegment packedK;
                MemorySegment packedV;
                if (sharedKvState != null) {
                    packedK = sharedKvState.packedKeyData();
                    packedV = sharedKvState.packedValueData();
                } else {
                    kContiguous = k.contiguous();
                    vContiguous = v.contiguous();
                    long blockElements = (long) maxBlocks * numKVHeads * blockSize * headDim;
                    packedK = arena.allocate(batch * blockElements * Float.BYTES, 64);
                    packedV = arena.allocate(batch * blockElements * Float.BYTES, 64);
                    PagedKvCacheMaterializer.packDenseSharedKvIntoTemporaryPagedPool(kContiguous, vContiguous,
                            numKVHeads, headDim, packedK, packedV);
                }

                MemorySegment blockTable = arena.allocate(batch * maxBlocks * Integer.BYTES, Integer.BYTES);
                for (int b = 0; b < batch; b++) {
                    blockTable.setAtIndex(ValueLayout.JAVA_INT, b * maxBlocks, b);
                }

                MemorySegment contextLens = arena.allocate(batch * Integer.BYTES, Integer.BYTES);
                for (int b = 0; b < batch; b++) {
                    contextLens.setAtIndex(ValueLayout.JAVA_INT, b, totalTokens);
                }

                int result = slidingLayer
                        ? (numKVHeads == numQHeads
                                ? binding.attentionWindowed(
                                        out.dataPtr(), qContiguous.dataPtr(), packedK, packedV,
                                        blockTable, contextLens,
                                        Math.toIntExact(batch), Math.toIntExact(seqLenQ), numQHeads, numKVHeads,
                                        headDim,
                                        blockSize, maxBlocks,
                                        scale, causal ? 1 : 0, startPos, config.slidingWindowSize(), softCap)
                                : binding.attentionGqaWindowed(
                                        out.dataPtr(), qContiguous.dataPtr(), packedK, packedV,
                                        blockTable, contextLens,
                                        Math.toIntExact(batch), Math.toIntExact(seqLenQ), numQHeads, numKVHeads,
                                        headDim,
                                        blockSize, maxBlocks,
                                        scale, causal ? 1 : 0, startPos, config.slidingWindowSize(), softCap))
                        : (numKVHeads == numQHeads
                                ? binding.attention(
                                        out.dataPtr(), qContiguous.dataPtr(), packedK, packedV,
                                        blockTable, contextLens,
                                        Math.toIntExact(batch), Math.toIntExact(seqLenQ), numQHeads, headDim,
                                        blockSize, maxBlocks,
                                        scale, causal ? 1 : 0, softCap)
                                : binding.attentionGqa(
                                        out.dataPtr(), qContiguous.dataPtr(), packedK, packedV,
                                        blockTable, contextLens,
                                        Math.toIntExact(batch), Math.toIntExact(seqLenQ), numQHeads, numKVHeads,
                                        headDim,
                                        blockSize, maxBlocks,
                                        scale, causal ? 1 : 0, softCap));
                if (result == 0) {
                    DirectInferenceProfiler.recordAttentionPath(usePackedSharedDecode
                            ? "dense_shared_packed_metal"
                            : "dense_shared_legacy_metal");
                    return out;
                }
                out.close();
            }
        } catch (RuntimeException e) {
            if (out != null && !out.isClosed()) {
                out.close();
            }
            return null;
        } finally {
            if (qContiguous != null && qContiguous != q && !qContiguous.isClosed()) {
                qContiguous.close();
            }
            if (kContiguous != null && kContiguous != k && !kContiguous.isClosed()) {
                kContiguous.close();
            }
            if (vContiguous != null && vContiguous != v && !vContiguous.isClosed()) {
                vContiguous.close();
            }
        }
        return null;
    }

    private MetalBinding metalBinding() {
        return metalBinding.get();
    }

    private MetalFlashAttentionBinding metalFa4() {
        return metalFa4.get();
    }

    private FlashAttentionRoutingPolicy routingPolicy() {
        return routingPolicy.get();
    }
}
