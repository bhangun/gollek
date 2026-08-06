/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FlashAttentionRoutingPolicyTest {

    @Test
    void injectedOptionsControlLegacyMetalBridgeForRestrictedModels() throws Exception {
        FlashAttentionModelPolicy modelPolicy = FlashAttentionModelPolicy.resolve(null, restrictedConfig());

        assertFalse(policy(false).allowLegacyMetalAttentionBridge(modelPolicy));
        assertTrue(policy(true).allowLegacyMetalAttentionBridge(modelPolicy));
    }

    @Test
    void injectedPagedOptionsFlowThroughTopLevelRoutingPolicy() throws Exception {
        FlashAttentionRoutingPolicy policy = new FlashAttentionRoutingPolicy(
                () -> false,
                () -> null,
                () -> null,
                new FlashAttentionRoutingOptions(false,
                        FlashAttentionFa4RoutingOptions.defaults(),
                        FlashAttentionRestrictedMetalOptions.defaults(),
                        new FlashAttentionPagedRoutingOptions(true, false, false, false, false, 64, 1024, null)));
        FlashAttentionModelPolicy modelPolicy = FlashAttentionModelPolicy.resolve(null, restrictedConfig());

        assertTrue(policy.allowPagedMetalAttentionBridge(modelPolicy, 1, 4096));
        assertTrue(policy.preferPagedMetalAttentionBeforeFa4(modelPolicy, 1, 4096));
    }

    private static FlashAttentionRoutingPolicy policy(boolean legacyMetalBridgeEnabled) {
        return new FlashAttentionRoutingPolicy(
                () -> false,
                () -> null,
                () -> null,
                new FlashAttentionRoutingOptions(legacyMetalBridgeEnabled,
                        FlashAttentionFa4RoutingOptions.defaults(),
                        FlashAttentionRestrictedMetalOptions.defaults(),
                        FlashAttentionPagedRoutingOptions.defaults()));
    }

    private static ModelConfig restrictedConfig() throws Exception {
        return new ObjectMapper().readValue("""
                {
                  "model_type": "gemma4_text",
                  "hidden_size": 256,
                  "intermediate_size": 512,
                  "num_hidden_layers": 2,
                  "num_attention_heads": 4,
                  "num_key_value_heads": 2
                }
                """, ModelConfig.class);
    }
}
