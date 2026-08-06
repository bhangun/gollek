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

class FlashAttentionSlidingDecodeRoutingPolicyTest {

    @Test
    void allowsGemma4SlidingDecodeBridgeOnlyForDecodeShape() throws Exception {
        ModelConfig config = slidingGemma4Config();
        FlashAttentionModelPolicy modelPolicy = FlashAttentionModelPolicy.resolve(null, config);
        FlashAttentionSlidingDecodeRoutingPolicy policy = policy(true, false);

        assertTrue(policy.allowBridge(config, modelPolicy, 0, 1));
        assertFalse(policy.allowBridge(config, modelPolicy, 0, 2));
        assertFalse(policy.allowBridge(config, modelPolicy, 1, 1));
        assertFalse(policy.allowBridge(null, modelPolicy, 0, 1));
    }

    @Test
    void legacyBridgeAllowsSlidingLayerBeforeDecodeShapeGate() throws Exception {
        ModelConfig config = slidingGemma4Config();
        FlashAttentionModelPolicy modelPolicy = FlashAttentionModelPolicy.resolve(null, config);
        FlashAttentionSlidingDecodeRoutingPolicy policy = policy(true, true);

        assertTrue(policy.allowBridge(config, modelPolicy, 0, 4));
        assertFalse(policy.canUseMetalAttention(config, modelPolicy, 0, 4));
    }

    @Test
    void metalAttentionRequiresGlobalMetalAndRuntimeBinding() throws Exception {
        ModelConfig config = slidingGemma4Config();
        FlashAttentionModelPolicy modelPolicy = FlashAttentionModelPolicy.resolve(null, config);

        assertFalse(policy(false, false).canUseMetalAttention(config, modelPolicy, 0, 1));
        assertFalse(policy(true, false).canUseMetalAttention(config, modelPolicy, 0, 1));
    }

    private static FlashAttentionSlidingDecodeRoutingPolicy policy(boolean canUseMetal, boolean legacyBridge) {
        FlashAttentionRestrictedMetalRoutingPolicy restrictedRouting =
                new FlashAttentionRestrictedMetalRoutingPolicy(() -> null);
        return new FlashAttentionSlidingDecodeRoutingPolicy(
                () -> canUseMetal,
                () -> null,
                restrictedRouting,
                legacyBridge);
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
}
