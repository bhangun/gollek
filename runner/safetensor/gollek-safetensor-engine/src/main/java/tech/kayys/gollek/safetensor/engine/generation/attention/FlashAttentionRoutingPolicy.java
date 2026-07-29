/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import tech.kayys.alkhawarizm.metal.binding.MetalBinding;
import tech.kayys.alkhawarizm.metal.binding.MetalFlashAttentionBinding;
import tech.kayys.gollek.spi.model.ModelConfig;

import java.util.function.BooleanSupplier;
import java.util.function.Supplier;

final class FlashAttentionRoutingPolicy {
    private final FlashAttentionRoutingOptions options;
    private final FlashAttentionFa4RoutingPolicy fa4Routing;
    private final FlashAttentionRestrictedMetalRoutingPolicy restrictedMetalRouting;
    private final FlashAttentionMetalRoutingPolicy metalRouting;
    private final FlashAttentionSlidingDecodeRoutingPolicy slidingDecodeRouting;
    private final FlashAttentionPagedRoutingPolicy pagedRouting;

    FlashAttentionRoutingPolicy(BooleanSupplier canUseMetal, Supplier<MetalBinding> metalBinding,
            Supplier<MetalFlashAttentionBinding> metalFa4) {
        this(canUseMetal, metalBinding, metalFa4, FlashAttentionRoutingOptions.fromSystemPropertiesAndEnvironment());
    }

    FlashAttentionRoutingPolicy(BooleanSupplier canUseMetal, Supplier<MetalBinding> metalBinding,
            Supplier<MetalFlashAttentionBinding> metalFa4, FlashAttentionRoutingOptions options) {
        this.options = options;
        this.fa4Routing = new FlashAttentionFa4RoutingPolicy(canUseMetal, metalFa4, options.fa4Options());
        this.restrictedMetalRouting = new FlashAttentionRestrictedMetalRoutingPolicy(
                metalBinding, options.restrictedMetalOptions());
        this.metalRouting = new FlashAttentionMetalRoutingPolicy(canUseMetal, metalBinding, restrictedMetalRouting);
        this.slidingDecodeRouting = new FlashAttentionSlidingDecodeRoutingPolicy(
                canUseMetal, metalBinding, restrictedMetalRouting, options.legacyMetalAttentionBridgeEnabled());
        this.pagedRouting = new FlashAttentionPagedRoutingPolicy(
                options.legacyMetalAttentionBridgeEnabled(), options.pagedRoutingOptions());
    }

    boolean canUseMetalAttention(ModelConfig config, FlashAttentionModelPolicy modelPolicy, int layerIdx,
            int seqLen, int startPos, float softCap) {
        return metalRouting.canUseAttention(config, modelPolicy, layerIdx, seqLen, startPos,
                allowLegacyMetalAttentionBridge(modelPolicy), () -> canUseFa4Attention(softCap));
    }

    boolean canUseRestrictedSlidingPrefillFa4Attention(ModelConfig config, int layerIdx,
            int seqLen, int startPos, float softCap) {
        return restrictedMetalRouting.canUseSlidingPrefillFa4Attention(
                config, layerIdx, seqLen, startPos, canUseFa4Attention(softCap));
    }

    boolean canUseFa4PagedAttention(ModelConfig config, int layerIdx, float softCap) {
        return fa4Routing.canUsePagedAttention(config, layerIdx, softCap);
    }

    boolean canUseSlidingDecodeMetalAttention(ModelConfig config, FlashAttentionModelPolicy modelPolicy,
            int layerIdx, int seqLen) {
        return slidingDecodeRouting.canUseMetalAttention(config, modelPolicy, layerIdx, seqLen);
    }

    boolean allowSlidingDecodeMetalAttentionBridge(ModelConfig config, FlashAttentionModelPolicy modelPolicy,
            int layerIdx, int seqLen) {
        return slidingDecodeRouting.allowBridge(config, modelPolicy, layerIdx, seqLen);
    }

    boolean canUseDenseRestrictedAttention(ModelConfig config, FlashAttentionModelPolicy modelPolicy, int layerIdx) {
        return restrictedMetalRouting.canUseDenseAttention(config, modelPolicy, layerIdx);
    }

    boolean canUseFa4Attention(float softCap) {
        return fa4Routing.canUseAttention(softCap);
    }

    boolean preferPagedMetalAttentionBeforeFa4(FlashAttentionModelPolicy modelPolicy, int seqLen, int totalTokens) {
        return pagedRouting.preferPagedMetalAttentionBeforeFa4(modelPolicy, seqLen, totalTokens);
    }

    boolean allowPagedMetalAttentionBridge(FlashAttentionModelPolicy modelPolicy, int seqLen, int totalTokens) {
        return pagedRouting.allowPagedMetalAttentionBridge(modelPolicy, seqLen, totalTokens);
    }

    boolean allowLegacyMetalAttentionBridge(FlashAttentionModelPolicy modelPolicy) {
        return modelPolicy.allowLegacyMetalAttentionBridge(options.legacyMetalAttentionBridgeEnabled());
    }

    boolean shouldUsePackedSharedDecodeAttention(ModelConfig config, FlashAttentionModelPolicy modelPolicy,
            long seqLenQ, SharedKvState sharedKvState) {
        return restrictedMetalRouting.shouldUsePackedSharedDecodeAttention(modelPolicy, seqLenQ, sharedKvState);
    }

    boolean shortDecodeUsesNativeAttention(int totalTokens) {
        return pagedRouting.shortDecodeUsesNativeAttention(totalTokens);
    }

    boolean enableRawPagedSlidingDecodeAttention() {
        return pagedRouting.enableRawPagedSlidingDecodeAttention();
    }

    String pagedAttentionPathName(boolean slidingLayer, long seqLen, int totalTokens, boolean firstChoice) {
        return pagedRouting.pathName(slidingLayer, seqLen, totalTokens, firstChoice);
    }
}
