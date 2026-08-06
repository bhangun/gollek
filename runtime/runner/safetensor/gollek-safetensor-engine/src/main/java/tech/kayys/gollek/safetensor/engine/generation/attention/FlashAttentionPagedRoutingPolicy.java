/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

final class FlashAttentionPagedRoutingPolicy {
    private static final int DEFAULT_PREFER_PAGED_METAL_PREFILL_MAX_TOKENS = 0;

    private final boolean legacyMetalAttentionBridgeEnabled;
    private final FlashAttentionPagedRoutingOptions options;

    FlashAttentionPagedRoutingPolicy(boolean legacyMetalAttentionBridgeEnabled) {
        this(legacyMetalAttentionBridgeEnabled, FlashAttentionPagedRoutingOptions.fromSystemPropertiesAndEnvironment());
    }

    FlashAttentionPagedRoutingPolicy(boolean legacyMetalAttentionBridgeEnabled,
            FlashAttentionPagedRoutingOptions options) {
        this.legacyMetalAttentionBridgeEnabled = legacyMetalAttentionBridgeEnabled;
        this.options = options == null ? FlashAttentionPagedRoutingOptions.defaults() : options;
    }

    boolean preferPagedMetalAttentionBeforeFa4(FlashAttentionModelPolicy modelPolicy, int seqLen, int totalTokens) {
        if (!allowPagedMetalAttentionBridge(modelPolicy, seqLen, totalTokens)) {
            return false;
        }
        if (seqLen == 1 && allowRestrictedPagedDecode(totalTokens)) {
            return true;
        }
        int prefillMaxTokens = pagedMetalPrefillMaxTokens(modelPolicy);
        return prefillMaxTokens > 0
                && seqLen > 1
                && totalTokens <= prefillMaxTokens;
    }

    boolean allowPagedMetalAttentionBridge(FlashAttentionModelPolicy modelPolicy, int seqLen, int totalTokens) {
        if (allowLegacyMetalAttentionBridge(modelPolicy)) {
            return true;
        }
        if (!modelPolicy.restrictedMetalDecodeCandidate(seqLen)) {
            return false;
        }
        return allowRestrictedPagedDecode(totalTokens);
    }

    boolean shortDecodeUsesNativeAttention(int totalTokens) {
        if (options.forceDecodeAttentionKernel()) {
            return false;
        }
        if (options.disableDecodeAttentionKernel()) {
            return true;
        }
        return totalTokens > 0 && totalTokens < options.decodeAttentionGpuMinContext();
    }

    boolean enableRawPagedSlidingDecodeAttention() {
        return options.enableRawPagedSlidingDecodeAttention();
    }

    String pathName(boolean slidingLayer, long seqLen, int totalTokens, boolean firstChoice) {
        String phase;
        if (seqLen == 1) {
            phase = shortDecodeUsesNativeAttention(totalTokens) ? "native_decode" : "metal_decode";
        } else {
            phase = "native_prefill";
        }
        String window = slidingLayer ? "_windowed" : "";
        return "paged_" + phase + window + (firstChoice ? "_first" : "");
    }

    private boolean allowLegacyMetalAttentionBridge(FlashAttentionModelPolicy modelPolicy) {
        return modelPolicy.allowLegacyMetalAttentionBridge(legacyMetalAttentionBridgeEnabled);
    }

    private boolean allowRestrictedPagedDecode(int totalTokens) {
        if (options.disableRestrictedPagedDecodeAttention()) {
            return false;
        }
        if (options.enableRestrictedPagedDecodeAttention()) {
            return true;
        }
        int maxTokens = options.preferPagedMetalAttentionMaxTokens();
        return maxTokens > 0 && totalTokens <= maxTokens;
    }

    private int pagedMetalPrefillMaxTokens(FlashAttentionModelPolicy modelPolicy) {
        String rawValue = options.preferPagedMetalPrefillMaxTokensValue();
        if (rawValue != null && !rawValue.isBlank()) {
            try {
                return Math.max(0, Integer.parseInt(rawValue.trim()));
            } catch (NumberFormatException ignored) {
                return DEFAULT_PREFER_PAGED_METAL_PREFILL_MAX_TOKENS;
            }
        }
        return modelPolicy.defaultPagedMetalPrefillMaxTokens(DEFAULT_PREFER_PAGED_METAL_PREFILL_MAX_TOKENS);
    }
}
