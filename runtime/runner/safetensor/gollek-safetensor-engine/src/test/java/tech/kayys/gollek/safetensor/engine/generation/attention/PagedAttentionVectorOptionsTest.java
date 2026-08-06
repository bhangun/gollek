/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PagedAttentionVectorOptionsTest {
    private static final String DEBUG_ATTENTION_PROBE_PROPERTY = "gollek.debug.attention_probe";

    @Test
    void defaultsDisableDebugAttentionProbe() {
        assertFalse(PagedAttentionVectorOptions.defaults().debugAttentionProbe());
    }

    @Test
    void systemPropertyControlsDebugAttentionProbe() {
        String previous = System.getProperty(DEBUG_ATTENTION_PROBE_PROPERTY);
        try {
            System.setProperty(DEBUG_ATTENTION_PROBE_PROPERTY, "true");

            assertTrue(PagedAttentionVectorOptions.fromSystemProperties().debugAttentionProbe());
        } finally {
            if (previous == null) {
                System.clearProperty(DEBUG_ATTENTION_PROBE_PROPERTY);
            } else {
                System.setProperty(DEBUG_ATTENTION_PROBE_PROPERTY, previous);
            }
        }
    }
}
