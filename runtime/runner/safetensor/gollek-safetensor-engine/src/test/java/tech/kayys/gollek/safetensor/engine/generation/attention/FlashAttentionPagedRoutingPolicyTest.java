/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FlashAttentionPagedRoutingPolicyTest {
    private static final String ENABLE_RESTRICTED_PAGED_DECODE_ATTENTION_PROPERTY =
            "gollek.safetensor.enable_gemma4_paged_decode_attention";
    private static final String PREFER_PAGED_METAL_ATTENTION_MAX_TOKENS_PROPERTY =
            "gollek.safetensor.prefer_paged_metal_attention_max_tokens";
    private static final String PREFER_PAGED_METAL_PREFILL_MAX_TOKENS_PROPERTY =
            "gollek.safetensor.prefer_paged_metal_prefill_max_tokens";

    @Test
    void gemma4PagedBridgeDefaultsToShortDecodeOnly() throws Exception {
        FlashAttentionPagedRoutingPolicy policy = defaultPolicy(false);
        FlashAttentionModelPolicy modelPolicy = FlashAttentionModelPolicy.resolve(null, gemma4Config());

        assertTrue(policy.allowPagedMetalAttentionBridge(modelPolicy, 1, 8));
        assertTrue(policy.preferPagedMetalAttentionBeforeFa4(modelPolicy, 1, 8));
        assertFalse(policy.allowPagedMetalAttentionBridge(modelPolicy, 2, 8));
    }

    @Test
    void legacyBridgeBypassesGemma4DecodeShapeRestriction() throws Exception {
        FlashAttentionPagedRoutingPolicy policy = defaultPolicy(true);
        FlashAttentionModelPolicy modelPolicy = FlashAttentionModelPolicy.resolve(null, gemma4Config());

        assertTrue(policy.allowPagedMetalAttentionBridge(modelPolicy, 4, 4096));
    }

    @Test
    void pathNamesIncludePhaseWindowAndFirstChoice() {
        FlashAttentionPagedRoutingPolicy policy = defaultPolicy(false);

        assertEquals("paged_native_prefill_windowed_first", policy.pathName(true, 4, 4, true));
        assertEquals("paged_native_prefill", policy.pathName(false, 4, 4, false));
    }

    @Test
    void injectedOptionsCanDisableRestrictedPagedDecode() throws Exception {
        FlashAttentionPagedRoutingPolicy policy = new FlashAttentionPagedRoutingPolicy(false,
                new FlashAttentionPagedRoutingOptions(false, true, false, false, false, 64, 1024, null));
        FlashAttentionModelPolicy modelPolicy = FlashAttentionModelPolicy.resolve(null, gemma4Config());

        assertFalse(policy.allowPagedMetalAttentionBridge(modelPolicy, 1, 8));
        assertFalse(policy.preferPagedMetalAttentionBeforeFa4(modelPolicy, 1, 8));
    }

    @Test
    void injectedOptionsCanEnableRestrictedPagedDecodePastDefaultTokenLimit() throws Exception {
        FlashAttentionPagedRoutingPolicy policy = new FlashAttentionPagedRoutingPolicy(false,
                new FlashAttentionPagedRoutingOptions(true, false, false, false, false, 64, 1024, null));
        FlashAttentionModelPolicy modelPolicy = FlashAttentionModelPolicy.resolve(null, gemma4Config());

        assertTrue(policy.allowPagedMetalAttentionBridge(modelPolicy, 1, 4096));
        assertTrue(policy.preferPagedMetalAttentionBeforeFa4(modelPolicy, 1, 4096));
    }

    @Test
    void injectedOptionsControlDecodeKernelAndRawSlidingDecodeGates() {
        FlashAttentionPagedRoutingPolicy forcedKernel = new FlashAttentionPagedRoutingPolicy(false,
                new FlashAttentionPagedRoutingOptions(false, false, true, true, false, 64, 1024, null));
        FlashAttentionPagedRoutingPolicy disabledKernel = new FlashAttentionPagedRoutingPolicy(false,
                new FlashAttentionPagedRoutingOptions(false, false, false, false, true, 64, 1024, null));

        assertFalse(forcedKernel.shortDecodeUsesNativeAttention(8));
        assertTrue(forcedKernel.enableRawPagedSlidingDecodeAttention());
        assertTrue(disabledKernel.shortDecodeUsesNativeAttention(4096));
        assertFalse(disabledKernel.enableRawPagedSlidingDecodeAttention());
    }

    @Test
    void injectedPrefillThresholdControlsPagedMetalPrefillPreference() throws Exception {
        FlashAttentionPagedRoutingPolicy policy = new FlashAttentionPagedRoutingPolicy(true,
                new FlashAttentionPagedRoutingOptions(false, false, false, false, false, 64, 1024, "16"));
        FlashAttentionModelPolicy modelPolicy = FlashAttentionModelPolicy.resolve(null, gemma4Config());

        assertTrue(policy.preferPagedMetalAttentionBeforeFa4(modelPolicy, 4, 16));
        assertFalse(policy.preferPagedMetalAttentionBeforeFa4(modelPolicy, 4, 17));
    }

    @Test
    void invalidInjectedPrefillThresholdDisablesPagedMetalPrefillPreference() throws Exception {
        FlashAttentionPagedRoutingPolicy policy = new FlashAttentionPagedRoutingPolicy(true,
                new FlashAttentionPagedRoutingOptions(false, false, false, false, false, 64, 1024, "not-a-number"));
        FlashAttentionModelPolicy modelPolicy = FlashAttentionModelPolicy.resolve(null, gemma4Config());

        assertFalse(policy.preferPagedMetalAttentionBeforeFa4(modelPolicy, 4, 1));
    }

    @Test
    void systemPropertiesControlRawPagedRoutingRequests() {
        String previousEnable = System.getProperty(ENABLE_RESTRICTED_PAGED_DECODE_ATTENTION_PROPERTY);
        String previousDecodeMax = System.getProperty(PREFER_PAGED_METAL_ATTENTION_MAX_TOKENS_PROPERTY);
        String previousPrefillMax = System.getProperty(PREFER_PAGED_METAL_PREFILL_MAX_TOKENS_PROPERTY);
        try {
            System.setProperty(ENABLE_RESTRICTED_PAGED_DECODE_ATTENTION_PROPERTY, "true");
            System.setProperty(PREFER_PAGED_METAL_ATTENTION_MAX_TOKENS_PROPERTY, "77");
            System.setProperty(PREFER_PAGED_METAL_PREFILL_MAX_TOKENS_PROPERTY, "33");

            FlashAttentionPagedRoutingOptions options =
                    FlashAttentionPagedRoutingOptions.fromSystemPropertiesAndEnvironment();

            assertTrue(options.enableRestrictedPagedDecodeAttention());
            assertEquals(77, options.preferPagedMetalAttentionMaxTokens());
            assertEquals("33", options.preferPagedMetalPrefillMaxTokensValue());
        } finally {
            restoreProperty(ENABLE_RESTRICTED_PAGED_DECODE_ATTENTION_PROPERTY, previousEnable);
            restoreProperty(PREFER_PAGED_METAL_ATTENTION_MAX_TOKENS_PROPERTY, previousDecodeMax);
            restoreProperty(PREFER_PAGED_METAL_PREFILL_MAX_TOKENS_PROPERTY, previousPrefillMax);
        }
    }

    private static FlashAttentionPagedRoutingPolicy defaultPolicy(boolean legacyMetalAttentionBridgeEnabled) {
        return new FlashAttentionPagedRoutingPolicy(
                legacyMetalAttentionBridgeEnabled, FlashAttentionPagedRoutingOptions.defaults());
    }

    private static ModelConfig gemma4Config() throws Exception {
        return new ObjectMapper().readValue("""
                {
                  "model_type": "gemma4_text",
                  "hidden_size": 256,
                  "intermediate_size": 512,
                  "num_hidden_layers": 4,
                  "num_attention_heads": 4,
                  "num_key_value_heads": 2
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
