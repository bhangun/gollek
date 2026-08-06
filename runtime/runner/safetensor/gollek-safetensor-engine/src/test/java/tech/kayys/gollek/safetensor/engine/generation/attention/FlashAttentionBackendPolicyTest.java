/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FlashAttentionBackendPolicyTest {
    private static final String FORCE_CPU_FORWARD_PROPERTY = "gollek.safetensor.force_cpu_forward";

    @Test
    void defaultsAllowReadyMetalBackend() {
        FlashAttentionBackendPolicy policy = FlashAttentionBackendPolicy.from(FlashAttentionBackendOptions.defaults());

        assertTrue(policy.canUseMetal(true));
        assertFalse(policy.canUseMetal(false));
    }

    @Test
    void forcedCpuDisablesReadyMetalBackend() {
        FlashAttentionBackendPolicy policy = FlashAttentionBackendPolicy.from(new FlashAttentionBackendOptions(true));

        assertFalse(policy.canUseMetal(true));
    }

    @Test
    void nullOptionsUseDefaults() {
        FlashAttentionBackendPolicy policy = FlashAttentionBackendPolicy.from(null);

        assertTrue(policy.canUseMetal(true));
    }

    @Test
    void systemPropertyControlsForcedCpuForwardRequest() {
        String previous = System.getProperty(FORCE_CPU_FORWARD_PROPERTY);
        try {
            System.setProperty(FORCE_CPU_FORWARD_PROPERTY, "true");

            FlashAttentionBackendOptions options = FlashAttentionBackendOptions.fromSystemProperties();

            assertTrue(options.forceCpuForward());
            assertFalse(FlashAttentionBackendPolicy.from(options).canUseMetal(true));
        } finally {
            if (previous == null) {
                System.clearProperty(FORCE_CPU_FORWARD_PROPERTY);
            } else {
                System.setProperty(FORCE_CPU_FORWARD_PROPERTY, previous);
            }
        }
    }
}
