/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import tech.kayys.alkhawarizm.metal.binding.MetalFlashAttentionBinding;
import tech.kayys.gollek.spi.model.ModelConfig;

import java.util.function.BooleanSupplier;
import java.util.function.Supplier;

final class FlashAttentionFa4RoutingPolicy {
    private final BooleanSupplier canUseMetal;
    private final Supplier<MetalFlashAttentionBinding> metalFa4;
    private final FlashAttentionFa4RoutingOptions options;

    FlashAttentionFa4RoutingPolicy(BooleanSupplier canUseMetal, Supplier<MetalFlashAttentionBinding> metalFa4) {
        this(canUseMetal, metalFa4, FlashAttentionFa4RoutingOptions.fromSystemProperties());
    }

    FlashAttentionFa4RoutingPolicy(BooleanSupplier canUseMetal, Supplier<MetalFlashAttentionBinding> metalFa4,
            FlashAttentionFa4RoutingOptions options) {
        this.canUseMetal = canUseMetal;
        this.metalFa4 = metalFa4;
        this.options = options;
    }

    boolean canUseAttention(float softCap) {
        return !options.disableFa4Attention()
                && nativeAvailable();
    }

    boolean canUsePagedAttention(ModelConfig config, int layerIdx, float softCap) {
        if (!canUseMetal.getAsBoolean() || !canUseAttention(softCap)) {
            return false;
        }
        if (config != null && config.usesSharedKvCache(layerIdx)) {
            return false;
        }
        if (config != null && config.isSlidingAttentionLayer(layerIdx) && config.hasSlidingWindow()) {
            return false;
        }
        return true;
    }

    private boolean nativeAvailable() {
        MetalFlashAttentionBinding binding = metalFa4.get();
        return binding != null
                && binding.isNativeAvailable();
    }
}
