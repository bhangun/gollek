/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertNotNull;

class FlashAttentionStageOptionsTest {

    @Test
    void defaultsProvideAllChildOptionGroups() {
        FlashAttentionStageOptions options = FlashAttentionStageOptions.defaults();

        assertNotNull(options.routingOptions());
        assertNotNull(options.ropeOptions());
        assertNotNull(options.normalizerOptions());
        assertNotNull(options.normalizationOptions());
        assertNotNull(options.precisionOptions());
        assertNotNull(options.pagedAttentionOptions());
        assertNotNull(options.backendOptions());
        assertNotNull(options.linearOptions());
        assertNotNull(options.matvecOptions());
    }

    @Test
    void compactConstructorDefaultsNullChildOptionGroups() {
        FlashAttentionStageOptions options = new FlashAttentionStageOptions(null, null, null, null);

        assertNotNull(options.routingOptions());
        assertNotNull(options.ropeOptions());
        assertNotNull(options.normalizerOptions());
        assertNotNull(options.normalizationOptions());
        assertNotNull(options.precisionOptions());
        assertNotNull(options.pagedAttentionOptions());
        assertNotNull(options.backendOptions());
        assertNotNull(options.linearOptions());
        assertNotNull(options.matvecOptions());
    }
}
