/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import tech.kayys.alkhawarizm.metal.binding.MetalBinding;
import tech.kayys.gollek.spi.model.ModelConfig;

import java.util.function.BooleanSupplier;
import java.util.function.Supplier;

final class FlashAttentionSlidingDecodeRoutingPolicy {
    private final BooleanSupplier canUseMetal;
    private final Supplier<MetalBinding> metalBinding;
    private final FlashAttentionRestrictedMetalRoutingPolicy restrictedMetalRouting;
    private final boolean legacyMetalAttentionBridgeEnabled;

    FlashAttentionSlidingDecodeRoutingPolicy(
            BooleanSupplier canUseMetal,
            Supplier<MetalBinding> metalBinding,
            FlashAttentionRestrictedMetalRoutingPolicy restrictedMetalRouting,
            boolean legacyMetalAttentionBridgeEnabled) {
        this.canUseMetal = canUseMetal;
        this.metalBinding = metalBinding;
        this.restrictedMetalRouting = restrictedMetalRouting;
        this.legacyMetalAttentionBridgeEnabled = legacyMetalAttentionBridgeEnabled;
    }

    boolean canUseMetalAttention(ModelConfig config, FlashAttentionModelPolicy modelPolicy,
            int layerIdx, int seqLen) {
        if (!canUseMetal.getAsBoolean()) {
            return false;
        }
        if (!allowBridge(config, modelPolicy, layerIdx, seqLen)) {
            return false;
        }
        if (seqLen != 1) {
            return false;
        }
        MetalBinding binding = metalBinding.get();
        if (binding == null || !binding.isRuntimeActive()) {
            return false;
        }
        if (!restrictedMetalRouting.slidingWindowFitsPackedMetal(config)) {
            return false;
        }
        if (restrictedMetalRouting.blocksGeneralMetalAttention(modelPolicy)) {
            return false;
        }
        return !config.usesSharedKvCache(layerIdx);
    }

    boolean allowBridge(ModelConfig config, FlashAttentionModelPolicy modelPolicy,
            int layerIdx, int seqLen) {
        if (config == null || !config.isSlidingAttentionLayer(layerIdx) || !config.hasSlidingWindow()) {
            return false;
        }
        if (allowLegacyMetalAttentionBridge(modelPolicy)) {
            return true;
        }
        if (!modelPolicy.restrictedMetalDecodeCandidate(seqLen)) {
            return false;
        }
        if (!restrictedMetalRouting.slidingWindowFitsPackedMetal(config)) {
            return false;
        }
        return restrictedMetalRouting.allowsMetalAttention();
    }

    private boolean allowLegacyMetalAttentionBridge(FlashAttentionModelPolicy modelPolicy) {
        return modelPolicy.allowLegacyMetalAttentionBridge(legacyMetalAttentionBridgeEnabled);
    }
}
