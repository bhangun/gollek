/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import org.junit.jupiter.api.Test;

import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertEquals;

class DirectForwardMetalHalfLinearExecutionPlanTest {

    @Test
    void nativeBf16MatvecRequiresSingleRowNativeWeightPolicyAndSymbol() {
        DirectForwardMetalHalfLinearExecutionPlan enabled =
                plan(1, true, allCapabilities(), true, false, false);
        DirectForwardMetalHalfLinearExecutionPlan multiRow =
                plan(2, true, allCapabilities(), true, false, false);
        DirectForwardMetalHalfLinearExecutionPlan nonNative =
                plan(1, false, allCapabilities(), true, false, false);
        DirectForwardMetalHalfLinearExecutionPlan missingSymbol =
                plan(1, true, DirectForwardMetalCapabilities.EMPTY, true, false, false);

        assertTrue(enabled.nativeBf16MatvecCandidate());
        assertFalse(multiRow.nativeBf16MatvecCandidate());
        assertFalse(nonNative.nativeBf16MatvecCandidate());
        assertFalse(missingSymbol.nativeBf16MatvecCandidate());
    }

    @Test
    void nonNativeMatvecCandidatesStaySeparateAndLabeled() {
        DirectForwardMetalHalfLinearExecutionPlan plan =
                plan(1, false, allCapabilities(), true, true, true);

        assertTrue(plan.mpsHalfMatvecCandidate());
        assertTrue(plan.transposedHalfMatvecCandidate());
        assertTrue(plan.halfMatvecCandidate());
        assertEquals("mps_matvec", plan.mpsHalfMatvecPath());
        assertEquals("transposed_matvec", plan.transposedHalfMatvecPath());
        assertEquals("matvec", plan.halfMatvecPath());
        assertEquals("metal_matmul", plan.matmulPath());
    }

    @Test
    void multiRowUsesOnlyMatmulFallback() {
        DirectForwardMetalHalfLinearExecutionPlan plan =
                plan(4, false, allCapabilities(), true, true, true);

        assertFalse(plan.nativeBf16MatvecCandidate());
        assertFalse(plan.mpsHalfMatvecCandidate());
        assertFalse(plan.transposedHalfMatvecCandidate());
        assertFalse(plan.halfMatvecCandidate());
        assertEquals("metal_matmul", plan.matmulPath());
    }

    @Test
    void transposedWeightMustMatchInputAndOutputDims() {
        DirectForwardMetalHalfLinearExecutionPlan plan =
                plan(1, false, allCapabilities(), false, false, true);
        AccelTensor matching = AccelTensor.zeros(3, 5);
        AccelTensor mismatched = AccelTensor.zeros(5, 3);

        try {
            assertTrue(plan.matchesTransposedWeight(matching));
            assertFalse(plan.matchesTransposedWeight(mismatched));
            assertFalse(plan.matchesTransposedWeight(null));
        } finally {
            matching.close();
            mismatched.close();
        }
    }

    private static DirectForwardMetalHalfLinearExecutionPlan plan(
            int rows,
            boolean nativeBf16Weight,
            DirectForwardMetalCapabilities capabilities,
            boolean halfMatvecAllowed,
            boolean mpsHalfMatvecAllowed,
            boolean transposedHalfMatvecAllowed) {
        return DirectForwardMetalHalfLinearExecutionPlan.from(
                rows,
                3,
                5,
                nativeBf16Weight,
                capabilities,
                halfMatvecAllowed,
                mpsHalfMatvecAllowed,
                transposedHalfMatvecAllowed);
    }

    private static DirectForwardMetalCapabilities allCapabilities() {
        return new DirectForwardMetalCapabilities(
                true, true, true, true, true, true, true, true, true, true,
                true, true, true, true, true, true, true, true, true, true,
                true, true, false, false);
    }
}
