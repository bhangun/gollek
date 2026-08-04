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

class DirectForwardMetalHalfLinearPairAdmissionPlanTest {

    @Test
    void admitsEligiblePairWhenFastPathPolicyAndSymbolAllowIt() {
        AccelTensor input = f32(1, 3);
        AccelTensor firstWeight = f16(2, 3);
        AccelTensor secondWeight = f16(2, 3);

        try {
            DirectForwardMetalHalfLinearPairAdmissionPlan plan =
                    plan(true, allCapabilities(), true, input, firstWeight, secondWeight);

            assertTrue(plan.admitted());
            assertNull(plan.rejectionDecision());
        } finally {
            input.close();
            firstWeight.close();
            secondWeight.close();
        }
    }

    @Test
    void rejectsWhenPairFastPathPolicyDisablesPair() {
        AccelTensor input = f32(1, 3);
        AccelTensor firstWeight = f16(2, 3);
        AccelTensor secondWeight = f16(2, 3);

        try {
            DirectForwardMetalHalfLinearPairAdmissionPlan plan =
                    plan(false, allCapabilities(), true, input, firstWeight, secondWeight);

            assertFalse(plan.admitted());
            assertEquals("reject:disabled", plan.rejectionDecision());
        } finally {
            input.close();
            firstWeight.close();
            secondWeight.close();
        }
    }

    @Test
    void rejectsFirstIneligibleCandidateBeforeSecondCandidate() {
        AccelTensor input = f32(1, 3);
        AccelTensor firstWeight = f32(2, 3);
        AccelTensor secondWeight = f16(2, 3);

        try {
            DirectForwardMetalHalfLinearPairAdmissionPlan plan =
                    plan(true, allCapabilities(), true, input, firstWeight, secondWeight);

            assertFalse(plan.admitted());
            assertEquals("reject:first_candidate_ineligible", plan.rejectionDecision());
        } finally {
            input.close();
            firstWeight.close();
            secondWeight.close();
        }
    }

    @Test
    void rejectsSecondIneligibleCandidateAfterFirstCandidatePasses() {
        AccelTensor input = f32(1, 3);
        AccelTensor firstWeight = f16(2, 3);
        AccelTensor secondWeight = f32(2, 3);

        try {
            DirectForwardMetalHalfLinearPairAdmissionPlan plan =
                    plan(true, allCapabilities(), true, input, firstWeight, secondWeight);

            assertFalse(plan.admitted());
            assertEquals("reject:second_candidate_ineligible", plan.rejectionDecision());
        } finally {
            input.close();
            firstWeight.close();
            secondWeight.close();
        }
    }

    @Test
    void rejectsWhenPairMatmulSymbolIsUnavailable() {
        AccelTensor input = f32(1, 3);
        AccelTensor firstWeight = f16(2, 3);
        AccelTensor secondWeight = f16(2, 3);

        try {
            DirectForwardMetalHalfLinearPairAdmissionPlan plan =
                    plan(true, DirectForwardMetalCapabilities.EMPTY, true, input, firstWeight, secondWeight);

            assertFalse(plan.admitted());
            assertEquals("reject:pair_symbol_unavailable", plan.rejectionDecision());
        } finally {
            input.close();
            firstWeight.close();
            secondWeight.close();
        }
    }

    @Test
    void rejectsWhenMetalLinearIsGloballyDisabled() {
        AccelTensor input = f32(1, 3);
        AccelTensor firstWeight = f16(2, 3);
        AccelTensor secondWeight = f16(2, 3);

        try {
            DirectForwardMetalHalfLinearPairAdmissionPlan plan =
                    plan(true, allCapabilities(), false, input, firstWeight, secondWeight);

            assertFalse(plan.admitted());
            assertEquals("reject:first_candidate_ineligible", plan.rejectionDecision());
        } finally {
            input.close();
            firstWeight.close();
            secondWeight.close();
        }
    }

    private static DirectForwardMetalHalfLinearPairAdmissionPlan plan(
            boolean pairFastPathAllowed,
            DirectForwardMetalCapabilities capabilities,
            boolean metalLinearEnabled,
            AccelTensor input,
            AccelTensor firstWeight,
            AccelTensor secondWeight) {
        return DirectForwardMetalHalfLinearPairAdmissionPlan.from(
                capabilities,
                generic(),
                metalLinearEnabled,
                input,
                firstWeight,
                secondWeight,
                pairFastPathAllowed);
    }

    private static AccelTensor f16(long... shape) {
        return AccelTensor.zeros(shape)
                .withQuantization(AccelTensor.QuantType.F16, null, null, -1);
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
