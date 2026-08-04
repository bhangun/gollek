/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.spi.model.FFNActivationType;

class DirectForwardMetalMatvecGatedFfnAdmissionPlanTest {

    @Test
    void admitsSiluForQwenWhenMetalIsAvailable() {
        DirectForwardMetalMatvecGatedFfnAdmissionPlan plan =
                plan(true, qwen(), true, FFNActivationType.SILU, null, null, null);

        assertTrue(plan.admitted());
        assertNull(plan.rejectionReason());
        assertNull(plan.rejectionDecision());
        assertTrue(plan.activation().silu());
    }

    @Test
    void admitsGeluForGemmaWhenMetalIsAvailable() {
        DirectForwardMetalMatvecGatedFfnAdmissionPlan plan =
                plan(true, gemma4(), true, FFNActivationType.GELU, null, null, null);

        assertTrue(plan.admitted());
        assertTrue(plan.activation().gelu());
    }

    @Test
    void rejectsUnsupportedActivationAfterFlagChecks() {
        DirectForwardMetalMatvecGatedFfnAdmissionPlan plan =
                plan(false, generic(), false, null, null, null, null);

        assertFalse(plan.admitted());
        assertEquals("unsupported_activation:null", plan.rejectionReason());
        assertEquals("reject:unsupported_activation:null", plan.rejectionDecision());
    }

    @Test
    void rejectsFlagDisabledForUnsupportedModelFamily() {
        DirectForwardMetalMatvecGatedFfnAdmissionPlan silu =
                plan(true, generic(), true, FFNActivationType.SILU, null, null, null);
        DirectForwardMetalMatvecGatedFfnAdmissionPlan gelu =
                plan(true, generic(), true, FFNActivationType.GELU, null, null, null);

        assertFalse(silu.admitted());
        assertEquals("swiglu_flag_disabled", silu.rejectionReason());
        assertFalse(gelu.admitted());
        assertEquals("geglu_flag_disabled", gelu.rejectionReason());
    }

    @Test
    void rejectsBiasBeforeMetalAvailability() {
        AccelTensor bias = f32(2);

        try {
            DirectForwardMetalMatvecGatedFfnAdmissionPlan plan =
                    plan(false, qwen(), false, FFNActivationType.SILU, bias, null, null);

            assertFalse(plan.admitted());
            assertEquals("bias_present", plan.rejectionReason());
        } finally {
            bias.close();
        }
    }

    @Test
    void rejectsMissingMetalAfterFlagAndBiasChecks() {
        DirectForwardMetalMatvecGatedFfnAdmissionPlan plan =
                plan(false, qwen(), true, FFNActivationType.SILU, null, null, null);

        assertFalse(plan.admitted());
        assertEquals("metal_unavailable", plan.rejectionReason());
    }

    @Test
    void shouldAttemptUsesAdmissionPlanForFamilyAndRowChecks() {
        AccelTensor singleRow = f32(1, 3);
        AccelTensor multiRow = f32(2, 3);

        try {
            assertTrue(DirectForwardMetalMatvecGatedFfn.shouldAttempt(singleRow, qwen(), FFNActivationType.SILU));
            assertTrue(DirectForwardMetalMatvecGatedFfn.shouldAttempt(singleRow, gemma4(), FFNActivationType.GELU));
            assertFalse(DirectForwardMetalMatvecGatedFfn.shouldAttempt(singleRow, generic(), FFNActivationType.SILU));
            assertFalse(DirectForwardMetalMatvecGatedFfn.shouldAttempt(multiRow, qwen(), FFNActivationType.SILU));
        } finally {
            singleRow.close();
            multiRow.close();
        }
    }

    private static DirectForwardMetalMatvecGatedFfnAdmissionPlan plan(
            boolean metalBindingAvailable,
            ModelConfigTraits traits,
            boolean metalLinearEnabled,
            FFNActivationType activationType,
            AccelTensor gateB,
            AccelTensor upB,
            AccelTensor downB) {
        return DirectForwardMetalMatvecGatedFfnAdmissionPlan.from(
                metalBindingAvailable,
                traits,
                metalLinearEnabled,
                activationType,
                gateB,
                upB,
                downB);
    }

    private static AccelTensor f32(long... shape) {
        return AccelTensor.zeros(shape);
    }

    private static ModelConfigTraits generic() {
        return traits("llama", false, false, false, false);
    }

    private static ModelConfigTraits gemma4() {
        return traits("gemma4_text", true, false, false, false);
    }

    private static ModelConfigTraits qwen() {
        return traits("qwen2", false, false, true, false);
    }

    private static ModelConfigTraits traits(
            String modelType,
            boolean gemma4Text,
            boolean gemma3Text,
            boolean qwenText,
            boolean gemma4StylePerLayerInputs) {
        return new ModelConfigTraits(
                null, modelType, 0, 0, gemma4Text, gemma3Text, qwenText, gemma4StylePerLayerInputs, false, false, false);
    }
}
