/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FlashAttentionRestrictedMetalRoutingPolicyTest {
    private static final String ALLOW_METAL_RESTRICTED_ATTENTION_PROPERTY =
            "gollek.safetensor.allow_metal_gemma4_attention";
    private static final String DISABLE_METAL_RESTRICTED_ATTENTION_PROPERTY =
            "gollek.safetensor.disable_metal_gemma4_attention";
    private static final String ENABLE_SHARED_DECODE_PACKED_ATTENTION_PROPERTY =
            "gollek.safetensor.enable_gemma4_shared_decode_packed_attention";
    private static final String ENABLE_SLIDING_PREFILL_FA4_ATTENTION_PROPERTY =
            "gollek.safetensor.enable_gemma4_sliding_prefill_fa4_attention";

    @Test
    void enablesSlidingPrefillFa4OnlyInsideSlidingPrefillWindow() throws Exception {
        FlashAttentionRestrictedMetalRoutingPolicy policy = defaultPolicy();
        ModelConfig config = slidingGemma4Config();

        assertTrue(policy.canUseSlidingPrefillFa4Attention(config, 0, 4, 0, true));
        assertFalse(policy.canUseSlidingPrefillFa4Attention(config, 0, 1, 0, true));
        assertFalse(policy.canUseSlidingPrefillFa4Attention(config, 0, 4, 509, true));
        assertFalse(policy.canUseSlidingPrefillFa4Attention(config, 1, 4, 0, true));
        assertFalse(policy.canUseSlidingPrefillFa4Attention(config, 0, 4, 0, false));
    }

    @Test
    void packedSharedDecodeRequiresRestrictedDecodeStateAndWindowedBinding() throws Exception {
        FlashAttentionRestrictedMetalRoutingPolicy policy = defaultPolicy();
        FlashAttentionModelPolicy modelPolicy = FlashAttentionModelPolicy.resolve(null, slidingGemma4Config());

        try (SharedKvState sharedKvState = sharedState()) {
            assertFalse(policy.shouldUsePackedSharedDecodeAttention(modelPolicy, 1, sharedKvState));
            assertFalse(policy.shouldUsePackedSharedDecodeAttention(modelPolicy, 2, sharedKvState));
            assertFalse(policy.shouldUsePackedSharedDecodeAttention(modelPolicy, 1, null));
        }
    }

    @Test
    void defaultRestrictedMetalGateAllowsGeneralMetalAttention() throws Exception {
        FlashAttentionRestrictedMetalRoutingPolicy policy = defaultPolicy();
        FlashAttentionModelPolicy modelPolicy = FlashAttentionModelPolicy.resolve(null, slidingGemma4Config());

        assertFalse(policy.blocksGeneralMetalAttention(modelPolicy));
        assertTrue(policy.allowsMetalAttention());
        assertTrue(policy.slidingWindowFitsPackedMetal(slidingGemma4Config()));
    }

    @Test
    void injectedOptionsCanDisableRestrictedMetalPathsWithoutSystemProperties() throws Exception {
        FlashAttentionRestrictedMetalRoutingPolicy policy = new FlashAttentionRestrictedMetalRoutingPolicy(
                () -> null,
                new FlashAttentionRestrictedMetalOptions(false, true, false, false, null, null, false));
        FlashAttentionModelPolicy modelPolicy = FlashAttentionModelPolicy.resolve(null, slidingGemma4Config());

        assertFalse(policy.canUseSlidingPrefillFa4Attention(slidingGemma4Config(), 0, 4, 0, true));
        try (SharedKvState sharedKvState = sharedState()) {
            assertFalse(policy.shouldUsePackedSharedDecodeAttention(modelPolicy, 1, sharedKvState));
        }
        assertTrue(policy.blocksGeneralMetalAttention(modelPolicy));
        assertFalse(policy.allowsMetalAttention());
    }

    @Test
    void injectedOptionsCanForceDenseRestrictedAttentionOnDenseLayers() throws Exception {
        FlashAttentionRestrictedMetalRoutingPolicy policy = new FlashAttentionRestrictedMetalRoutingPolicy(
                () -> null,
                new FlashAttentionRestrictedMetalOptions(false, false, false, false, null, null, true));
        FlashAttentionModelPolicy modelPolicy = FlashAttentionModelPolicy.resolve(null, slidingGemma4Config());

        assertFalse(policy.canUseDenseAttention(slidingGemma4Config(), modelPolicy, 0));
        assertTrue(policy.canUseDenseAttention(slidingGemma4Config(), modelPolicy, 1));
    }

    @Test
    void injectedExplicitFalseDisablesSlidingPrefillFa4Attention() throws Exception {
        FlashAttentionRestrictedMetalRoutingPolicy policy = new FlashAttentionRestrictedMetalRoutingPolicy(
                () -> null,
                new FlashAttentionRestrictedMetalOptions(false, false, false, false, null, "false", false));

        assertFalse(policy.canUseSlidingPrefillFa4Attention(slidingGemma4Config(), 0, 4, 0, true));
    }

    @Test
    void nullOptionsUseDefaults() throws Exception {
        FlashAttentionRestrictedMetalRoutingPolicy policy =
                new FlashAttentionRestrictedMetalRoutingPolicy(() -> null, null);

        assertTrue(policy.canUseSlidingPrefillFa4Attention(slidingGemma4Config(), 0, 4, 0, true));
    }

    @Test
    void systemPropertiesControlRawRestrictedMetalRequests() {
        String previousAllow = System.getProperty(ALLOW_METAL_RESTRICTED_ATTENTION_PROPERTY);
        String previousDisable = System.getProperty(DISABLE_METAL_RESTRICTED_ATTENTION_PROPERTY);
        String previousSharedDecode = System.getProperty(ENABLE_SHARED_DECODE_PACKED_ATTENTION_PROPERTY);
        String previousSlidingPrefill = System.getProperty(ENABLE_SLIDING_PREFILL_FA4_ATTENTION_PROPERTY);
        try {
            System.setProperty(ALLOW_METAL_RESTRICTED_ATTENTION_PROPERTY, "true");
            System.setProperty(DISABLE_METAL_RESTRICTED_ATTENTION_PROPERTY, "true");
            System.setProperty(ENABLE_SHARED_DECODE_PACKED_ATTENTION_PROPERTY, "false");
            System.setProperty(ENABLE_SLIDING_PREFILL_FA4_ATTENTION_PROPERTY, "true");

            FlashAttentionRestrictedMetalOptions options =
                    FlashAttentionRestrictedMetalOptions.fromSystemProperties();

            assertTrue(options.allowMetalRestrictedAttention());
            assertTrue(options.disableMetalRestrictedAttention());
            assertEquals("true", options.enableSlidingPrefillFa4AttentionValue());
            assertEquals("false", options.enableSharedDecodePackedAttentionValue());
        } finally {
            restoreProperty(ALLOW_METAL_RESTRICTED_ATTENTION_PROPERTY, previousAllow);
            restoreProperty(DISABLE_METAL_RESTRICTED_ATTENTION_PROPERTY, previousDisable);
            restoreProperty(ENABLE_SHARED_DECODE_PACKED_ATTENTION_PROPERTY, previousSharedDecode);
            restoreProperty(ENABLE_SLIDING_PREFILL_FA4_ATTENTION_PROPERTY, previousSlidingPrefill);
        }
    }

    private static SharedKvState sharedState() {
        return new SharedKvState(AccelTensor.zeros(1, 1, 1, 1), AccelTensor.zeros(1, 1, 1, 1));
    }

    private static FlashAttentionRestrictedMetalRoutingPolicy defaultPolicy() {
        return new FlashAttentionRestrictedMetalRoutingPolicy(
                () -> null, FlashAttentionRestrictedMetalOptions.defaults());
    }

    private static ModelConfig slidingGemma4Config() throws Exception {
        return new ObjectMapper().readValue("""
                {
                  "model_type": "gemma4_text",
                  "hidden_size": 256,
                  "intermediate_size": 512,
                  "num_hidden_layers": 2,
                  "num_attention_heads": 4,
                  "num_key_value_heads": 2,
                  "sliding_window": 512,
                  "layer_types": ["sliding_attention", "full_attention"]
                }
                """, ModelConfig.class);
    }

    private static void restoreProperty(String name, String previous) {
        if (previous == null) {
            System.clearProperty(name);
        } else {
            System.setProperty(name, previous);
        }
    }
}
