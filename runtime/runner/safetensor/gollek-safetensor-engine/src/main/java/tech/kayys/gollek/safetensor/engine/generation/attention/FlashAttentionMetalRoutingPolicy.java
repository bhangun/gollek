/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import tech.kayys.alkhawarizm.metal.binding.MetalBinding;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;

import java.util.function.BooleanSupplier;
import java.util.function.Supplier;

final class FlashAttentionMetalRoutingPolicy {
    private final BooleanSupplier canUseMetal;
    private final Supplier<MetalBinding> metalBinding;
    private final FlashAttentionRestrictedMetalRoutingPolicy restrictedMetalRouting;

    FlashAttentionMetalRoutingPolicy(BooleanSupplier canUseMetal, Supplier<MetalBinding> metalBinding,
            FlashAttentionRestrictedMetalRoutingPolicy restrictedMetalRouting) {
        this.canUseMetal = canUseMetal;
        this.metalBinding = metalBinding;
        this.restrictedMetalRouting = restrictedMetalRouting;
    }

    boolean canUseAttention(ModelConfig config, FlashAttentionModelPolicy modelPolicy, int layerIdx,
            int seqLen, int startPos, boolean legacyBridgeAllowed, BooleanSupplier fa4Available) {
        if (!canUseMetal.getAsBoolean()) {
            return false;
        }
        if (!legacyBridgeAllowed && modelPolicy.restrictsLegacyMetalAttentionBridge()) {
            return restrictedMetalRouting.canUseSlidingPrefillFa4Attention(
                    config, layerIdx, seqLen, startPos, fa4Available.getAsBoolean());
        }
        if (config != null && config.usesSharedKvCache(layerIdx)) {
            return false;
        }
        if (config != null && config.isSlidingAttentionLayer(layerIdx) && config.hasSlidingWindow()) {
            MetalBinding binding = metalBinding.get();
            return binding != null && binding.isWindowedAttentionAvailable();
        }
        if (restrictedMetalRouting.blocksGeneralMetalAttention(modelPolicy)) {
            return false;
        }
        return true;
    }
}
