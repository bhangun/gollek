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

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.alkhawarizm.spi.model.FFNActivationType;

class DirectForwardMetalFusedGatedFfnAdmissionPlanTest {

    @Test
    void rejectsSiluWhenDecodeNotOptimizedForModel() {
        AccelTensor input = f32(1, 3);

        try {
            DirectForwardMetalFusedGatedFfnAdmissionPlan plan =
                    plan(true, allCapabilities(), true, input, FFNActivationType.SILU, null, null, null);
            assertFalse(plan.admitted());
            assertEquals("decode_not_optimized_for_model", plan.rejectionReason());
        } finally {
            input.close();
        }
    }

    @Test
    void rejectsUnsupportedActivationBeforeOtherChecks() {
        DirectForwardMetalFusedGatedFfnAdmissionPlan plan =
                plan(false, DirectForwardMetalCapabilities.EMPTY, false, null, null, null, null, null);

        assertFalse(plan.admitted());
        assertEquals("unsupported_activation:null", plan.rejectionReason());
        assertEquals("reject:unsupported_activation:null", plan.rejectionDecision());
    }

    @Test
    void rejectsBiasBeforeMetalAvailability() {
        AccelTensor input = f32(1, 3);
        AccelTensor bias = f32(2);

        try {
            DirectForwardMetalFusedGatedFfnAdmissionPlan plan =
                    plan(false, allCapabilities(), false, input, FFNActivationType.SILU, bias, null, null);

            assertFalse(plan.admitted());
            assertEquals("bias_present", plan.rejectionReason());
        } finally {
            input.close();
            bias.close();
        }
    }

    @Test
    void rejectsMetalLinearDisabledBeforeMissingBinding() {
        AccelTensor input = f32(1, 3);

        try {
            DirectForwardMetalFusedGatedFfnAdmissionPlan plan =
                    plan(false, allCapabilities(), false, input, FFNActivationType.SILU, null, null, null);

            assertFalse(plan.admitted());
            assertEquals("metal_linear_disabled", plan.rejectionReason());
        } finally {
            input.close();
        }
    }

    @Test
    void rejectsMissingBindingBeforeCapabilityCheck() {
        AccelTensor input = f32(1, 3);

        try {
            DirectForwardMetalFusedGatedFfnAdmissionPlan plan =
                    plan(false, DirectForwardMetalCapabilities.EMPTY, true, input, FFNActivationType.SILU,
                            null, null, null);

            assertFalse(plan.admitted());
            assertEquals("metal_binding_unavailable", plan.rejectionReason());
        } finally {
            input.close();
        }
    }

    @Test
    void rejectsNullInputBeforeMissingCapability() {
        DirectForwardMetalFusedGatedFfnAdmissionPlan plan =
                plan(true, DirectForwardMetalCapabilities.EMPTY, true, null, FFNActivationType.SILU,
                        null, null, null);

        assertFalse(plan.admitted());
        assertEquals("input_null", plan.rejectionReason());
    }

    @Test
    void rejectsNullInputAfterCapabilitiesPass() {
        DirectForwardMetalFusedGatedFfnAdmissionPlan plan =
                plan(true, allCapabilities(), true, null, FFNActivationType.SILU, null, null, null);

        assertFalse(plan.admitted());
        assertEquals("input_null", plan.rejectionReason());
    }

    @Test
    void skipReasonUsesAdmissionPlanRejectionReason() {
        AccelTensor input = f32(1, 3);

        try {
            assertEquals("metal_binding_unavailable", DirectForwardMetalFusedGatedFfn.skipReason(
                    null,
                    allCapabilities(),
                    generic(),
                    true,
                    input,
                    FFNActivationType.SILU,
                    null,
                    null,
                    null));
        } finally {
            input.close();
        }
    }

    @Test
    void rejectsInvalidRows() {
        AccelTensor input = f32(0, 3);

        try {
            DirectForwardMetalFusedGatedFfnAdmissionPlan plan =
                    plan(true, allCapabilities(), true, input, FFNActivationType.SILU, null, null, null);

            assertFalse(plan.admitted());
            assertEquals("invalid_rows:0", plan.rejectionReason());
            assertEquals(0L, plan.rows());
        } finally {
            input.close();
        }
    }

    private static DirectForwardMetalFusedGatedFfnAdmissionPlan plan(
            boolean metalBindingAvailable,
            DirectForwardMetalCapabilities capabilities,
            boolean metalLinearEnabled,
            AccelTensor input,
            FFNActivationType activationType,
            AccelTensor gateB,
            AccelTensor upB,
            AccelTensor downB) {
        return DirectForwardMetalFusedGatedFfnAdmissionPlan.from(
                metalBindingAvailable,
                capabilities,
                generic(),
                metalLinearEnabled,
                input,
                activationType,
                gateB,
                upB,
                downB);
    }

    private static AccelTensor f32(long... shape) {
        return AccelTensor.zeros(shape);
    }

    private static ModelConfigTraits generic() {
        return new ModelConfigTraits(null, "llama", 0, 0, false, false, false, false, false, false, false);
    }

    private static DirectForwardMetalCapabilities allCapabilities() {
        return new DirectForwardMetalCapabilities(
                true, true, true, true, true, true, true, true, true, true,
                true, true, true, true, true, true, true, true, true, true,
                true, true, false, false);
    }
}
