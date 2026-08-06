/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import org.junit.jupiter.api.Test;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DirectForwardMetalFfnWeightPlanTest {
    private static final String PROFILE_KEY = "ffn_fused_metal";

    @Test
    void f16GatedPlanKeepsWeightsAndReportsUniformHalf() {
        AccelTensor gate = f16(2, 3);
        AccelTensor up = f16(2, 3);
        AccelTensor down = f16(3, 2);

        try {
            DirectForwardMetalFfnWeightPlan plan =
                    DirectForwardMetalFfnWeightPlan.gated(generic(), PROFILE_KEY, false, 1, gate, up, down);

            assertTrue(plan.completeGated());
            assertFalse(plan.nativeBf16Weights());
            assertTrue(plan.hasUniformHalfOrBf16Weights());
            assertNull(plan.gatedConversionFailureReason());
            assertNull(plan.gatedUniformTypeFailureReason());
            assertSame(gate, plan.gateW());
            assertSame(up, plan.upW());
            assertSame(down, plan.downW());
        } finally {
            gate.close();
            up.close();
            down.close();
        }
    }

    @Test
    void nativeBf16GatedPlanKeepsNativeBf16Weights() {
        AccelTensor gate = bf16(2, 3);
        AccelTensor up = bf16(2, 3);
        AccelTensor down = bf16(3, 2);

        try {
            DirectForwardMetalFfnWeightPlan plan =
                    DirectForwardMetalFfnWeightPlan.gated(gemma4(), PROFILE_KEY, false, 1, gate, up, down);

            assertTrue(plan.completeGated());
            assertTrue(plan.nativeBf16Weights());
            assertTrue(plan.hasUniformHalfOrBf16Weights());
            assertNull(plan.gatedConversionFailureReason());
            assertNull(plan.gatedUniformTypeFailureReason());
            assertSame(gate, plan.gateW());
            assertSame(up, plan.upW());
            assertSame(down, plan.downW());
        } finally {
            gate.close();
            up.close();
            down.close();
        }
    }

    @Test
    void gateUpPlanRejectsUnconvertedMixedHalfWeights() {
        AccelTensor gate = f16(2, 3);
        AccelTensor up = bf16(2, 3);

        try {
            DirectForwardMetalFfnWeightPlan plan =
                    DirectForwardMetalFfnWeightPlan.gateUp(gemma4(), gate, up, false, false);

            assertFalse(plan.completeGateUp());
            assertFalse(plan.hasUniformHalfOrBf16Weights());
            assertEquals("weight_conversion_failed:native_bf16=false", plan.gateUpConversionFailureReason());
        } finally {
            gate.close();
            up.close();
        }
    }

    @Test
    void gatedPlanReportsConversionFailureWhenAWeightIsMissing() {
        AccelTensor gate = f16(2, 3);
        AccelTensor up = f16(2, 3);

        try {
            DirectForwardMetalFfnWeightPlan plan =
                    new DirectForwardMetalFfnWeightPlan(gate, up, null, true, false);

            assertFalse(plan.completeGated());
            assertEquals("weight_conversion_failed:native_bf16=true", plan.gatedConversionFailureReason());
        } finally {
            gate.close();
            up.close();
        }
    }

    @Test
    void gatedPlanReportsTypeMismatchAfterCompleteMixedWeights() {
        AccelTensor gate = f16(2, 3);
        AccelTensor up = bf16(2, 3);
        AccelTensor down = f16(3, 2);

        try {
            DirectForwardMetalFfnWeightPlan plan =
                    new DirectForwardMetalFfnWeightPlan(gate, up, down, false, false);

            assertTrue(plan.completeGated());
            assertNull(plan.gatedConversionFailureReason());
            assertFalse(plan.hasUniformHalfOrBf16Weights());
            assertEquals("weight_type_mismatch:native_bf16=false", plan.gatedUniformTypeFailureReason());
        } finally {
            gate.close();
            up.close();
            down.close();
        }
    }

    private static AccelTensor f16(long... shape) {
        return AccelTensor.zeros(shape)
                .withQuantization(AccelTensor.QuantType.F16, null, null, -1);
    }

    private static AccelTensor bf16(long... shape) {
        return AccelTensor.zeros(shape)
                .withQuantization(AccelTensor.QuantType.BF16, null, null, -1);
    }

    private static ModelConfigTraits generic() {
        return traits(false);
    }

    private static ModelConfigTraits gemma4() {
        return traits(true);
    }

    private static ModelConfigTraits traits(boolean nativeBf16Matvec) {
        return new ModelConfigTraits(null, nativeBf16Matvec ? "gemma4_text" : "llama",
                0, 0, nativeBf16Matvec, false, false, false, false, false, false);
    }
}
