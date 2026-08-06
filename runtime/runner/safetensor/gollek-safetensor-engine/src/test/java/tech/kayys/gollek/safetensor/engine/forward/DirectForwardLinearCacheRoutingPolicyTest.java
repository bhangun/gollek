/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DirectForwardLinearCacheRoutingPolicyTest {

    @Test
    void ffnDownCacheRequiresSupportedProfileAndSingleTokenCandidate() {
        DirectForwardLinearCacheRoutingPolicy policy = policy(
                DirectForwardLinearCacheOptions.defaults().withFfnDownLargeHalfCacheBudgets(100, 10));

        assertTrue(policy.shouldCacheFfnDownHalfWeight("ffn_down", true, 10, 10));
        assertTrue(policy.shouldCacheFfnDownHalfWeight("ffn_down_nongated", true, 10, 10));
        assertFalse(policy.shouldCacheFfnDownHalfWeight("logits", true, 10, 10));
        assertFalse(policy.shouldCacheFfnDownHalfWeight("ffn_down", false, 10, 10));
    }

    @Test
    void ffnDownCacheHonorsPerTensorAndTotalBudgets() {
        DirectForwardLinearCacheOptions options =
                DirectForwardLinearCacheOptions.defaults().withFfnDownLargeHalfCacheBudgets(100, 10);
        DirectForwardLinearCacheRoutingPolicy policy = policy(options);

        assertTrue(policy.shouldCacheFfnDownHalfWeight("ffn_down", true, 10, 10));
        assertFalse(policy.shouldCacheFfnDownHalfWeight("ffn_down", true, 11, 10));
        assertFalse(policy.shouldCacheFfnDownHalfWeight("ffn_down", true, 10, 11));
        assertFalse(policy(options.withFfnDownLargeHalfCacheBudgets(0, 10))
                .shouldCacheFfnDownHalfWeight("ffn_down", true, 10, 10));
        assertFalse(policy(options.withFfnDownLargeHalfCacheBudgets(100, 0))
                .shouldCacheFfnDownHalfWeight("ffn_down", true, 10, 10));
    }

    @Test
    void ffnDownCacheTreatsOverflowAsOverBudget() {
        DirectForwardLinearCacheRoutingPolicy policy = policy(
                DirectForwardLinearCacheOptions.defaults().withFfnDownLargeHalfCacheBudgets(Long.MAX_VALUE - 1, 10));

        assertFalse(policy.shouldCacheFfnDownHalfWeight("ffn_down", true, Long.MAX_VALUE, 2));
    }

    @Test
    void logitsLargeHalfCacheRequiresLogitsProfileAndSingleTokenCandidate() {
        DirectForwardLinearCacheRoutingPolicy policy = policy(
                DirectForwardLinearCacheOptions.defaults().withLogitsLargeHalfCacheMaxBytes(10));

        assertTrue(policy.shouldCacheLogitsLargeHalfWeight("logits", true, 10));
        assertFalse(policy.shouldCacheLogitsLargeHalfWeight("q_proj", true, 10));
        assertFalse(policy.shouldCacheLogitsLargeHalfWeight("logits", false, 10));
    }

    @Test
    void logitsLargeHalfCacheHonorsMaxBytes() {
        DirectForwardLinearCacheOptions options =
                DirectForwardLinearCacheOptions.defaults().withLogitsLargeHalfCacheMaxBytes(10);
        DirectForwardLinearCacheRoutingPolicy policy = policy(options);

        assertTrue(policy.shouldCacheLogitsLargeHalfWeight("logits", true, 10));
        assertFalse(policy.shouldCacheLogitsLargeHalfWeight("logits", true, 11));
        assertFalse(policy(options.withLogitsLargeHalfCacheMaxBytes(0))
                .shouldCacheLogitsLargeHalfWeight("logits", true, 1));
    }

    @Test
    void profileHelpersMirrorCacheProfiles() {
        DirectForwardLinearCacheRoutingPolicy policy = policy(DirectForwardLinearCacheOptions.defaults());

        assertTrue(policy.isFfnDownHalfCacheProfile("ffn_down"));
        assertTrue(policy.isFfnDownHalfCacheProfile("ffn_down_nongated"));
        assertFalse(policy.isFfnDownHalfCacheProfile("logits"));
        assertTrue(policy.isLogitsLargeHalfCacheProfile("logits"));
        assertFalse(policy.isLogitsLargeHalfCacheProfile("ffn_down"));
    }

    private static DirectForwardLinearCacheRoutingPolicy policy(DirectForwardLinearCacheOptions options) {
        return DirectForwardLinearCacheRoutingPolicy.from(options);
    }
}
