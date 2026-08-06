/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

record FlashAttentionRoutingOptions(
        boolean legacyMetalAttentionBridgeEnabled,
        FlashAttentionFa4RoutingOptions fa4Options,
        FlashAttentionRestrictedMetalOptions restrictedMetalOptions,
        FlashAttentionPagedRoutingOptions pagedRoutingOptions) {
    private static final String ALLOW_LEGACY_METAL_ATTENTION_BRIDGE_PROPERTY =
            "gollek.safetensor.allow_legacy_metal_attention_bridge";

    FlashAttentionRoutingOptions {
        if (fa4Options == null) {
            fa4Options = FlashAttentionFa4RoutingOptions.defaults();
        }
        if (restrictedMetalOptions == null) {
            restrictedMetalOptions = FlashAttentionRestrictedMetalOptions.defaults();
        }
        if (pagedRoutingOptions == null) {
            pagedRoutingOptions = FlashAttentionPagedRoutingOptions.defaults();
        }
    }

    static FlashAttentionRoutingOptions fromSystemPropertiesAndEnvironment() {
        return new FlashAttentionRoutingOptions(
                Boolean.getBoolean(ALLOW_LEGACY_METAL_ATTENTION_BRIDGE_PROPERTY),
                FlashAttentionFa4RoutingOptions.fromSystemProperties(),
                FlashAttentionRestrictedMetalOptions.fromSystemProperties(),
                FlashAttentionPagedRoutingOptions.fromSystemPropertiesAndEnvironment());
    }

    static FlashAttentionRoutingOptions defaults() {
        return new FlashAttentionRoutingOptions(false,
                FlashAttentionFa4RoutingOptions.defaults(),
                FlashAttentionRestrictedMetalOptions.defaults(),
                FlashAttentionPagedRoutingOptions.defaults());
    }
}
