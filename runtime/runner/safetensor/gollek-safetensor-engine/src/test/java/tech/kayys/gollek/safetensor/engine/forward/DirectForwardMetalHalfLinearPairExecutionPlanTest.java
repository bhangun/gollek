/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class DirectForwardMetalHalfLinearPairExecutionPlanTest {

    @Test
    void nativeBf16PairMatvecRequiresSingleRowNativeWeightsPolicyAndSymbol() {
        DirectForwardMetalHalfLinearPairExecutionPlan enabled =
                plan(1, true, allCapabilities(), true);
        DirectForwardMetalHalfLinearPairExecutionPlan multiRow =
                plan(2, true, allCapabilities(), true);
        DirectForwardMetalHalfLinearPairExecutionPlan nonNative =
                plan(1, false, allCapabilities(), true);
        DirectForwardMetalHalfLinearPairExecutionPlan missingSymbol =
                plan(1, true, DirectForwardMetalCapabilities.EMPTY, true);
        DirectForwardMetalHalfLinearPairExecutionPlan policyDisabled =
                plan(1, true, allCapabilities(), false);

        assertTrue(enabled.nativeBf16MatvecCandidate());
        assertFalse(multiRow.nativeBf16MatvecCandidate());
        assertFalse(nonNative.nativeBf16MatvecCandidate());
        assertFalse(missingSymbol.nativeBf16MatvecCandidate());
        assertFalse(policyDisabled.nativeBf16MatvecCandidate());
    }

    @Test
    void halfPairMatvecRequiresSingleRowNonNativeWeightsPolicyAndSymbol() {
        DirectForwardMetalHalfLinearPairExecutionPlan enabled =
                plan(1, false, allCapabilities(), true);
        DirectForwardMetalHalfLinearPairExecutionPlan multiRow =
                plan(2, false, allCapabilities(), true);
        DirectForwardMetalHalfLinearPairExecutionPlan nativeWeights =
                plan(1, true, allCapabilities(), true);
        DirectForwardMetalHalfLinearPairExecutionPlan missingSymbol =
                plan(1, false, DirectForwardMetalCapabilities.EMPTY, true);
        DirectForwardMetalHalfLinearPairExecutionPlan policyDisabled =
                plan(1, false, allCapabilities(), false);

        assertTrue(enabled.halfMatvecCandidate());
        assertFalse(multiRow.halfMatvecCandidate());
        assertFalse(nativeWeights.halfMatvecCandidate());
        assertFalse(missingSymbol.halfMatvecCandidate());
        assertFalse(policyDisabled.halfMatvecCandidate());
    }

    @Test
    void multiRowUsesOnlyMatmulFallbackAndStableLabels() {
        DirectForwardMetalHalfLinearPairExecutionPlan plan =
                plan(4, false, allCapabilities(), true);

        assertFalse(plan.nativeBf16MatvecCandidate());
        assertFalse(plan.halfMatvecCandidate());
        assertEquals("matvec", plan.halfMatvecPath());
        assertEquals("metal_pair_matmul", plan.matmulPath());
    }

    @Test
    void nativeBf16PathUsesPairDescriptor() {
        DirectForwardMetalHalfLinearPairExecutionPlan plan =
                DirectForwardMetalHalfLinearPairExecutionPlan.from(
                        1,
                        4096,
                        4096,
                        true,
                        allCapabilities(),
                        true);

        assertEquals(
                DirectForwardNativeBf16MatvecPolicy.describeNativeBf16PairMatvecPath(4096, 4096),
                plan.nativeBf16MatvecPath());
    }

    private static DirectForwardMetalHalfLinearPairExecutionPlan plan(
            int rows,
            boolean nativeBf16Weights,
            DirectForwardMetalCapabilities capabilities,
            boolean pairMatvecAllowed) {
        return DirectForwardMetalHalfLinearPairExecutionPlan.from(
                rows,
                3,
                5,
                nativeBf16Weights,
                capabilities,
                pairMatvecAllowed);
    }

    private static DirectForwardMetalCapabilities allCapabilities() {
        return new DirectForwardMetalCapabilities(
                true, true, true, true, true, true, true, true, true, true,
                true, true, true, true, true, true, true, true, true, true,
                true, true, false, false);
    }
}
