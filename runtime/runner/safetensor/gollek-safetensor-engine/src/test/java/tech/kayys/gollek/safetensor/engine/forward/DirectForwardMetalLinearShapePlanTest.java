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

class DirectForwardMetalLinearShapePlanTest {

    @Test
    void singlePlanExtractsRowsDimsAndOutputShape() {
        AccelTensor input = AccelTensor.zeros(1, 2, 3);
        AccelTensor weight = AccelTensor.zeros(5, 3);

        try {
            DirectForwardMetalLinearShapePlan plan =
                    DirectForwardMetalLinearShapePlan.single(input, weight);

            assertNotNull(plan);
            assertFalse(plan.singleRow());
            assertArrayEquals(new long[] {1, 2, 5}, plan.outputShape());
            org.junit.jupiter.api.Assertions.assertEquals(3, plan.inputDim());
            org.junit.jupiter.api.Assertions.assertEquals(2, plan.rows());
            org.junit.jupiter.api.Assertions.assertEquals(5, plan.outputDim());
        } finally {
            input.close();
            weight.close();
        }
    }

    @Test
    void singlePlanRejectsInputHiddenMismatch() {
        AccelTensor input = AccelTensor.zeros(1, 1, 3);
        AccelTensor weight = AccelTensor.zeros(5, 4);

        try {
            assertNull(DirectForwardMetalLinearShapePlan.single(input, weight));
        } finally {
            input.close();
            weight.close();
        }
    }

    @Test
    void singlePlanRejectsRankOneInput() {
        AccelTensor input = AccelTensor.zeros(3);
        AccelTensor weight = AccelTensor.zeros(5, 3);

        try {
            assertNull(DirectForwardMetalLinearShapePlan.single(input, weight));
        } finally {
            input.close();
            weight.close();
        }
    }

    @Test
    void pairPlanRequiresMatchingWeightsAndInputHidden() {
        AccelTensor input = AccelTensor.zeros(1, 1, 3);
        AccelTensor first = AccelTensor.zeros(5, 3);
        AccelTensor second = AccelTensor.zeros(5, 3);

        try {
            DirectForwardMetalLinearShapePlan plan =
                    DirectForwardMetalLinearShapePlan.pair(input, first, second);

            assertNotNull(plan);
            assertTrue(plan.singleRow());
            assertArrayEquals(new long[] {1, 1, 5}, plan.outputShape());
        } finally {
            input.close();
            first.close();
            second.close();
        }
    }

    @Test
    void pairPlanRejectsMismatchedSecondWeight() {
        AccelTensor input = AccelTensor.zeros(1, 1, 3);
        AccelTensor first = AccelTensor.zeros(5, 3);
        AccelTensor second = AccelTensor.zeros(4, 3);

        try {
            assertNull(DirectForwardMetalLinearShapePlan.pair(input, first, second));
        } finally {
            input.close();
            first.close();
            second.close();
        }
    }
}
