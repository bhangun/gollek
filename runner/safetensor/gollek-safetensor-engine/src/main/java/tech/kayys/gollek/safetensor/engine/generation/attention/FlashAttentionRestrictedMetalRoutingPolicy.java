/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import tech.kayys.alkhawarizm.metal.binding.MetalBinding;
import tech.kayys.gollek.spi.model.ModelConfig;

import java.util.function.Supplier;

final class FlashAttentionRestrictedMetalRoutingPolicy {
    private static final int MAX_PACKED_METAL_SLIDING_WINDOW = 2048;

    private final Supplier<MetalBinding> metalBinding;
    private final FlashAttentionRestrictedMetalOptions options;

    FlashAttentionRestrictedMetalRoutingPolicy(Supplier<MetalBinding> metalBinding) {
        this(metalBinding, FlashAttentionRestrictedMetalOptions.fromSystemProperties());
    }

    FlashAttentionRestrictedMetalRoutingPolicy(Supplier<MetalBinding> metalBinding,
            FlashAttentionRestrictedMetalOptions options) {
        this.metalBinding = metalBinding;
        this.options = options == null ? FlashAttentionRestrictedMetalOptions.defaults() : options;
    }

    boolean canUseSlidingPrefillFa4Attention(ModelConfig config, int layerIdx,
            int seqLen, int startPos, boolean fa4Available) {
        if (config == null || !config.isSlidingAttentionLayer(layerIdx) || !config.hasSlidingWindow()) {
            return false;
        }
        if (seqLen <= 1 || config.usesSharedKvCache(layerIdx)) {
            return false;
        }
        if (!slidingPrefillFa4AttentionEnabled()) {
            return false;
        }
        int totalTokens = startPos + seqLen;
        return totalTokens <= config.getSlidingWindowSize() && fa4Available;
    }

    boolean canUseDenseAttention(ModelConfig config, FlashAttentionModelPolicy modelPolicy, int layerIdx) {
        if (!modelPolicy.supportsForcedDenseAttention() || !options.forceDenseRestrictedAttention()) {
            return false;
        }
        return config == null || !config.isSlidingAttentionLayer(layerIdx) || !config.hasSlidingWindow();
    }

    boolean shouldUsePackedSharedDecodeAttention(FlashAttentionModelPolicy modelPolicy,
            long seqLenQ, SharedKvState sharedKvState) {
        if (sharedKvState == null || seqLenQ != 1L) {
            return false;
        }
        if (!modelPolicy.restrictsLegacyMetalAttentionBridge()
                || !sharedDecodePackedAttentionEnabled()) {
            return false;
        }
        MetalBinding binding = metalBinding.get();
        return binding != null
                && binding.isWindowedAttentionAvailable();
    }

    boolean blocksGeneralMetalAttention(FlashAttentionModelPolicy modelPolicy) {
        return modelPolicy.restrictsLegacyMetalAttentionBridge()
                && blocksGeneralMetalAttention();
    }

    boolean allowsMetalAttention() {
        return allowsRestrictedMetalAttention();
    }

    boolean slidingWindowFitsPackedMetal(ModelConfig config) {
        return config != null && config.getSlidingWindowSize() <= MAX_PACKED_METAL_SLIDING_WINDOW;
    }

    private boolean slidingPrefillFa4AttentionEnabled() {
        return !options.disableSlidingPrefillFa4Attention()
                && !restrictedMetalDisabled()
                && explicitFlagEnabled(options.enableSlidingPrefillFa4AttentionValue());
    }

    private boolean sharedDecodePackedAttentionEnabled() {
        return !options.disableSharedDecodePackedAttention()
                && !restrictedMetalDisabled()
                && explicitFlagEnabled(options.enableSharedDecodePackedAttentionValue());
    }

    private boolean blocksGeneralMetalAttention() {
        return options.disableMetalRestrictedAttention()
                && !options.allowMetalRestrictedAttention();
    }

    private boolean allowsRestrictedMetalAttention() {
        return !options.disableMetalRestrictedAttention()
                || options.allowMetalRestrictedAttention();
    }

    private boolean restrictedMetalDisabled() {
        return options.disableMetalRestrictedAttention();
    }

    private static boolean explicitFlagEnabled(String value) {
        return value == null || value.isBlank() || Boolean.parseBoolean(value);
    }
}
