/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.planning;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import tech.kayys.alkhawarizm.spi.model.ModelRuntimeTraits;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DirectForwardPlatformPolicyTest {
    private String previousForceCpuForward;
    private String previousForceCpuGemma4;
    private String previousAllowMetalGemma4;

    @BeforeEach
    void captureProperties() {
        previousForceCpuForward = System.getProperty(DirectForwardPlatformPolicy.FORCE_CPU_FORWARD_PROPERTY);
        previousForceCpuGemma4 = System.getProperty(DirectForwardPlatformPolicy.FORCE_CPU_GEMMA4_PROPERTY);
        previousAllowMetalGemma4 = System.getProperty(DirectForwardPlatformPolicy.ALLOW_METAL_GEMMA4_PROPERTY);
    }

    @AfterEach
    void restoreProperties() {
        restoreProperty(DirectForwardPlatformPolicy.FORCE_CPU_FORWARD_PROPERTY, previousForceCpuForward);
        restoreProperty(DirectForwardPlatformPolicy.FORCE_CPU_GEMMA4_PROPERTY, previousForceCpuGemma4);
        restoreProperty(DirectForwardPlatformPolicy.ALLOW_METAL_GEMMA4_PROPERTY, previousAllowMetalGemma4);
    }

    @Test
    void forcedGemma4CpuSetsForwardOverride() {
        System.setProperty(DirectForwardPlatformPolicy.FORCE_CPU_GEMMA4_PROPERTY, "true");

        DirectForwardPlatformPolicy.Decision decision =
                DirectForwardPlatformPolicy.apply(gemma4Traits(), "Gemma4ForCausalLM");

        assertTrue(decision.forceCpuForward());
        assertFalse(decision.experimentalMetalAllowed());
        assertTrue(Boolean.getBoolean(DirectForwardPlatformPolicy.FORCE_CPU_FORWARD_PROPERTY));
        assertTrue(decision.reason().contains(DirectForwardPlatformPolicy.FORCE_CPU_GEMMA4_PROPERTY));
    }

    @Test
    void experimentalMetalGemma4ClearsStaleCpuOverride() {
        System.setProperty(DirectForwardPlatformPolicy.FORCE_CPU_FORWARD_PROPERTY, "true");
        System.setProperty(DirectForwardPlatformPolicy.ALLOW_METAL_GEMMA4_PROPERTY, "true");

        DirectForwardPlatformPolicy.Decision decision =
                DirectForwardPlatformPolicy.apply(gemma4Traits(), "Gemma4ForCausalLM");

        assertFalse(decision.forceCpuForward());
        assertTrue(decision.experimentalMetalAllowed());
        assertNull(System.getProperty(DirectForwardPlatformPolicy.FORCE_CPU_FORWARD_PROPERTY));
    }

    @Test
    void nonGemmaTraitsClearStaleCpuOverrideEvenWhenGemmaFlagIsSet() {
        System.setProperty(DirectForwardPlatformPolicy.FORCE_CPU_FORWARD_PROPERTY, "true");
        System.setProperty(DirectForwardPlatformPolicy.FORCE_CPU_GEMMA4_PROPERTY, "true");

        DirectForwardPlatformPolicy.Decision decision =
                DirectForwardPlatformPolicy.apply(ModelRuntimeTraits.EMPTY, "GenericForCausalLM");

        assertFalse(decision.forceCpuForward());
        assertFalse(decision.experimentalMetalAllowed());
        assertNull(System.getProperty(DirectForwardPlatformPolicy.FORCE_CPU_FORWARD_PROPERTY));
    }

    private static ModelRuntimeTraits gemma4Traits() {
        return ModelRuntimeTraits.builder().nativeBf16Matvec().build();
    }

    private static void restoreProperty(String name, String previous) {
        if (previous == null) {
            System.clearProperty(name);
        } else {
            System.setProperty(name, previous);
        }
    }
}
