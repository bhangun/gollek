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

class DirectForwardMetalFfnShapeAdmissionPlanTest {

    @Test
    void gatedAdmissionAllowsMultiTokenRows() {
        AccelTensor input = f32(1, 2, 3);
        AccelTensor gate = f32(5, 3);
        AccelTensor up = f32(5, 3);
        AccelTensor down = f32(7, 5);

        try {
            DirectForwardMetalFfnShapeAdmissionPlan plan =
                    DirectForwardMetalFfnShapeAdmissionPlan.gated(input, gate, up, down);

            assertTrue(plan.admitted());
            assertNull(plan.rejectionReason());
            assertNull(plan.rejectionDecision());
            assertEquals(2L, plan.shapePlan().rows());
        } finally {
            input.close();
            gate.close();
            up.close();
            down.close();
        }
    }

    @Test
    void gatedAdmissionRejectsShapeMismatch() {
        AccelTensor input = f32(1, 1, 3);
        AccelTensor gate = f32(5, 3);
        AccelTensor up = f32(4, 3);
        AccelTensor down = f32(7, 5);

        try {
            DirectForwardMetalFfnShapeAdmissionPlan plan =
                    DirectForwardMetalFfnShapeAdmissionPlan.gated(input, gate, up, down);

            assertFalse(plan.admitted());
            assertNull(plan.shapePlan());
            assertEquals("shape_mismatch", plan.rejectionReason());
            assertEquals("reject:shape_mismatch", plan.rejectionDecision());
        } finally {
            input.close();
            gate.close();
            up.close();
            down.close();
        }
    }

    @Test
    void singleTokenGatedRejectsMultiTokenRows() {
        AccelTensor input = f32(1, 2, 3);
        AccelTensor gate = f32(5, 3);
        AccelTensor up = f32(5, 3);
        AccelTensor down = f32(7, 5);

        try {
            DirectForwardMetalFfnShapeAdmissionPlan plan =
                    DirectForwardMetalFfnShapeAdmissionPlan.singleTokenGated(input, gate, up, down);

            assertFalse(plan.admitted());
            assertEquals("not_single_token_rows:2", plan.rejectionReason());
            assertEquals("reject:not_single_token_rows:2", plan.rejectionDecision());
            assertEquals(2L, plan.shapePlan().rows());
        } finally {
            input.close();
            gate.close();
            up.close();
            down.close();
        }
    }

    @Test
    void singleTokenGateUpRequiresCombinedWorkspaceShape() {
        AccelTensor input = f32(1, 1, 3);
        AccelTensor gate = f32(5, 3);
        AccelTensor up = f32(5, 3);
        AccelTensor combined = f32(1, 1, 4);

        try {
            DirectForwardMetalFfnShapeAdmissionPlan plan =
                    DirectForwardMetalFfnShapeAdmissionPlan.singleTokenGateUp(input, gate, up, combined);

            assertFalse(plan.admitted());
            assertEquals("combined_shape_mismatch", plan.rejectionReason());
            assertEquals("reject:combined_shape_mismatch", plan.rejectionDecision());
            assertEquals(5L, plan.shapePlan().intermediateDim());
        } finally {
            input.close();
            gate.close();
            up.close();
            combined.close();
        }
    }

    @Test
    void singleTokenGateUpAdmitsMatchingCombinedWorkspace() {
        AccelTensor input = f32(1, 1, 3);
        AccelTensor gate = f32(5, 3);
        AccelTensor up = f32(5, 3);
        AccelTensor combined = f32(1, 1, 5);

        try {
            DirectForwardMetalFfnShapeAdmissionPlan plan =
                    DirectForwardMetalFfnShapeAdmissionPlan.singleTokenGateUp(input, gate, up, combined);

            assertTrue(plan.admitted());
            assertNull(plan.rejectionReason());
            assertEquals(5L, plan.shapePlan().intermediateDim());
        } finally {
            input.close();
            gate.close();
            up.close();
            combined.close();
        }
    }

    private static AccelTensor f32(long... shape) {
        return AccelTensor.zeros(shape);
    }
}
