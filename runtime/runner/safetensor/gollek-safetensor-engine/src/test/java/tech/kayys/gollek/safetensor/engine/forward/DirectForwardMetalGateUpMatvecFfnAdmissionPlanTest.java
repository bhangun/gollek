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

class DirectForwardMetalGateUpMatvecFfnAdmissionPlanTest {
    private static final String GATE_UP_PROPERTY =
            DirectForwardFfnFastPathOptions.ENABLE_METAL_GATE_UP_MATVEC_FFN_PROPERTY;

    @Test
    void admitsSupportedActivationWhenFlagMetalAndWorkspaceAreAvailable() {
        AccelTensor combined = f32(1, 4);

        try {
            DirectForwardMetalGateUpMatvecFfnAdmissionPlan plan = withGateUpFlag(true,
                    () -> plan(true, true, FFNActivationType.SILU, null, null, combined));

            assertTrue(plan.admitted());
            assertNull(plan.rejectionReason());
            assertNull(plan.rejectionDecision());
            assertTrue(plan.activation().silu());
        } finally {
            combined.close();
        }
    }

    @Test
    void rejectsFlagDisabledBeforeUnsupportedActivation() {
        AccelTensor combined = f32(1, 4);

        try {
            DirectForwardMetalGateUpMatvecFfnAdmissionPlan plan = withGateUpFlag(false,
                    () -> plan(true, true, null, null, null, combined));

            assertFalse(plan.admitted());
            assertEquals("flag_disabled", plan.rejectionReason());
            assertEquals("reject:flag_disabled", plan.rejectionDecision());
        } finally {
            combined.close();
        }
    }

    @Test
    void rejectsUnsupportedActivationAfterFlagEnabled() {
        AccelTensor combined = f32(1, 4);

        try {
            DirectForwardMetalGateUpMatvecFfnAdmissionPlan plan = withGateUpFlag(true,
                    () -> plan(true, true, null, null, null, combined));

            assertFalse(plan.admitted());
            assertEquals("unsupported_activation:null", plan.rejectionReason());
        } finally {
            combined.close();
        }
    }

    @Test
    void rejectsBiasBeforeWorkspaceAvailability() {
        AccelTensor bias = f32(4);

        try {
            DirectForwardMetalGateUpMatvecFfnAdmissionPlan plan = withGateUpFlag(true,
                    () -> plan(false, false, FFNActivationType.GELU, bias, null, null));

            assertFalse(plan.admitted());
            assertEquals("bias_present", plan.rejectionReason());
        } finally {
            bias.close();
        }
    }

    @Test
    void rejectsMissingWorkspaceBeforeMetalAvailability() {
        DirectForwardMetalGateUpMatvecFfnAdmissionPlan plan = withGateUpFlag(true,
                () -> plan(false, false, FFNActivationType.SILU, null, null, null));

        assertFalse(plan.admitted());
        assertEquals("combined_workspace_unavailable", plan.rejectionReason());
    }

    @Test
    void rejectsClosedWorkspaceBeforeMetalAvailability() {
        AccelTensor combined = f32(1, 4);
        combined.close();

        DirectForwardMetalGateUpMatvecFfnAdmissionPlan plan = withGateUpFlag(true,
                () -> plan(false, false, FFNActivationType.SILU, null, null, combined));

        assertFalse(plan.admitted());
        assertEquals("combined_workspace_unavailable", plan.rejectionReason());
    }

    @Test
    void rejectsMissingMetalAfterWorkspacePasses() {
        AccelTensor combined = f32(1, 4);

        try {
            DirectForwardMetalGateUpMatvecFfnAdmissionPlan plan = withGateUpFlag(true,
                    () -> plan(false, true, FFNActivationType.SILU, null, null, combined));

            assertFalse(plan.admitted());
            assertEquals("metal_unavailable", plan.rejectionReason());
        } finally {
            combined.close();
        }
    }

    private static DirectForwardMetalGateUpMatvecFfnAdmissionPlan plan(
            boolean metalBindingAvailable,
            boolean metalLinearEnabled,
            FFNActivationType activationType,
            AccelTensor gateB,
            AccelTensor upB,
            AccelTensor combinedBuffer) {
        return DirectForwardMetalGateUpMatvecFfnAdmissionPlan.from(
                metalBindingAvailable,
                metalLinearEnabled,
                activationType,
                gateB,
                upB,
                combinedBuffer);
    }

    private static DirectForwardMetalGateUpMatvecFfnAdmissionPlan withGateUpFlag(
            boolean enabled,
            PlanSupplier supplier) {
        String previous = System.getProperty(GATE_UP_PROPERTY);
        System.setProperty(GATE_UP_PROPERTY, Boolean.toString(enabled));
        try {
            return supplier.get();
        } finally {
            restoreProperty(previous);
        }
    }

    private static void restoreProperty(String previous) {
        if (previous == null) {
            System.clearProperty(GATE_UP_PROPERTY);
        } else {
            System.setProperty(GATE_UP_PROPERTY, previous);
        }
    }

    private static AccelTensor f32(long... shape) {
        return AccelTensor.zeros(shape);
    }

    private interface PlanSupplier {
        DirectForwardMetalGateUpMatvecFfnAdmissionPlan get();
    }
}
