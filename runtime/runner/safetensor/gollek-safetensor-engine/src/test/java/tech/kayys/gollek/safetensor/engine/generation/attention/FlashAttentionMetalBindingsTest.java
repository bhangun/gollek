/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FlashAttentionMetalBindingsTest {

    @Test
    void canUseMetalReflectsReadinessWhenCpuIsNotForced() {
        FlashAttentionMetalBindings bindings = new FlashAttentionMetalBindings(
                null, null, true, FlashAttentionBackendOptions.defaults());

        assertTrue(bindings.canUseMetal());
    }

    @Test
    void canUseMetalRejectsUnreadyOrForcedCpuBackends() {
        FlashAttentionMetalBindings unready = new FlashAttentionMetalBindings(
                null, null, false, FlashAttentionBackendOptions.defaults());
        FlashAttentionMetalBindings forcedCpu = new FlashAttentionMetalBindings(
                null, null, true, new FlashAttentionBackendOptions(true));

        assertFalse(unready.canUseMetal());
        assertFalse(forcedCpu.canUseMetal());
    }
}
