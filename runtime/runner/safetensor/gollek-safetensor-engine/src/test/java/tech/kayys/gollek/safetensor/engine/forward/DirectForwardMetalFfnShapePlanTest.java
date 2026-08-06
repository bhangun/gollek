/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import org.junit.jupiter.api.Test;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DirectForwardMetalFfnShapePlanTest {

    @Test
    void gatedPlanExtractsRowsDimsAndOutputShape() {
        AccelTensor input = AccelTensor.zeros(1, 2, 3);
        AccelTensor gate = AccelTensor.zeros(5, 3);
        AccelTensor up = AccelTensor.zeros(5, 3);
        AccelTensor down = AccelTensor.zeros(7, 5);

        try {
            DirectForwardMetalFfnShapePlan plan =
                    DirectForwardMetalFfnShapePlan.gated(input, gate, up, down);

            assertNotNull(plan);
            assertFalse(plan.singleRow());
            assertTrue(plan.validRows());
            assertArrayEquals(new long[] {1, 2, 7}, plan.outputShape());
            assertDims(plan, 3, 2, 5, 7);
        } finally {
            input.close();
            gate.close();
            up.close();
            down.close();
        }
    }

    @Test
    void gatedPlanRejectsMismatchedProjectionWeights() {
        AccelTensor input = AccelTensor.zeros(1, 1, 3);
        AccelTensor gate = AccelTensor.zeros(5, 3);
        AccelTensor up = AccelTensor.zeros(4, 3);
        AccelTensor down = AccelTensor.zeros(7, 5);

        try {
            assertNull(DirectForwardMetalFfnShapePlan.gated(input, gate, up, down));
        } finally {
            input.close();
            gate.close();
            up.close();
            down.close();
        }
    }

    @Test
    void gatedPlanRejectsInputDimMismatch() {
        AccelTensor input = AccelTensor.zeros(1, 1, 4);
        AccelTensor gate = AccelTensor.zeros(5, 3);
        AccelTensor up = AccelTensor.zeros(5, 3);
        AccelTensor down = AccelTensor.zeros(7, 5);

        try {
            assertNull(DirectForwardMetalFfnShapePlan.gated(input, gate, up, down));
        } finally {
            input.close();
            gate.close();
            up.close();
            down.close();
        }
    }

    @Test
    void gateUpPlanUsesIntermediateShapeForCombinedBuffer() {
        AccelTensor input = AccelTensor.zeros(1, 1, 3);
        AccelTensor gate = AccelTensor.zeros(5, 3);
        AccelTensor up = AccelTensor.zeros(5, 3);
        AccelTensor combined = AccelTensor.zeros(1, 1, 5);

        try {
            DirectForwardMetalFfnShapePlan plan =
                    DirectForwardMetalFfnShapePlan.gateUp(input, gate, up);

            assertNotNull(plan);
            assertTrue(plan.singleRow());
            assertTrue(plan.matchesOutputBuffer(combined));
            assertArrayEquals(new long[] {1, 1, 5}, plan.outputShape());
            assertDims(plan, 3, 1, 5, 5);
        } finally {
            input.close();
            gate.close();
            up.close();
            combined.close();
        }
    }

    private static void assertDims(DirectForwardMetalFfnShapePlan plan,
            long inputDim, long rows, long intermediateDim, long outputDim) {
        org.junit.jupiter.api.Assertions.assertEquals(inputDim, plan.inputDim());
        org.junit.jupiter.api.Assertions.assertEquals(rows, plan.rows());
        org.junit.jupiter.api.Assertions.assertEquals(intermediateDim, plan.intermediateDim());
        org.junit.jupiter.api.Assertions.assertEquals(outputDim, plan.outputDim());
    }
}
