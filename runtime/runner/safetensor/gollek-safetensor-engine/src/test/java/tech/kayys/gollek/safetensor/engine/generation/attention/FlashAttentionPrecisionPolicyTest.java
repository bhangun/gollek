/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FlashAttentionPrecisionPolicyTest {
    private static final String USE_BF16_ATTENTION_PROPERTY = "gollek.safetensor.use_bf16_attention";

    @Test
    void defaultsDoNotRequestBf16Attention() {
        FlashAttentionPrecisionOptions options = FlashAttentionPrecisionOptions.defaults();

        assertFalse(options.useBf16Attention());
        assertFalse(policy(options).useBf16Attention(null));
    }

    @Test
    void requestedBf16StillRequiresAvailableFa4Binding() {
        FlashAttentionPrecisionOptions options = new FlashAttentionPrecisionOptions(true);

        assertTrue(options.useBf16Attention());
        assertFalse(policy(options).useBf16Attention(null));
    }

    @Test
    void systemPropertyControlsBf16AttentionRequest() {
        String previous = System.getProperty(USE_BF16_ATTENTION_PROPERTY);
        try {
            System.setProperty(USE_BF16_ATTENTION_PROPERTY, "true");

            assertTrue(FlashAttentionPrecisionOptions.fromSystemProperties().useBf16Attention());
        } finally {
            if (previous == null) {
                System.clearProperty(USE_BF16_ATTENTION_PROPERTY);
            } else {
                System.setProperty(USE_BF16_ATTENTION_PROPERTY, previous);
            }
        }
    }

    private static FlashAttentionPrecisionPolicy policy(FlashAttentionPrecisionOptions options) {
        return FlashAttentionPrecisionPolicy.from(options);
    }
}
