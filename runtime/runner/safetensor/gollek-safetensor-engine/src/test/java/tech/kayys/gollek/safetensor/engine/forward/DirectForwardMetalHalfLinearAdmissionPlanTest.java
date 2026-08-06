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

class DirectForwardMetalHalfLinearAdmissionPlanTest {
    private static final String PROFILE_KEY = "ffn_down";

    @Test
    void admitsEligibleSingleLinearCandidate() {
        AccelTensor input = f32(1, 3);
        AccelTensor weight = f16(2, 3);

        try {
            DirectForwardMetalHalfLinearAdmissionPlan plan =
                    plan(true, input, weight);

            assertTrue(plan.admitted());
            assertNull(plan.rejectionDecision());
        } finally {
            input.close();
            weight.close();
        }
    }

    @Test
    void rejectsWhenMetalLinearIsDisabled() {
        AccelTensor input = f32(1, 3);
        AccelTensor weight = f16(2, 3);

        try {
            DirectForwardMetalHalfLinearAdmissionPlan plan =
                    plan(false, input, weight);

            assertFalse(plan.admitted());
            assertEquals("reject:candidate_ineligible", plan.rejectionDecision());
        } finally {
            input.close();
            weight.close();
        }
    }

    @Test
    void rejectsNonF32Input() {
        AccelTensor input = f16(1, 3);
        AccelTensor weight = f16(2, 3);

        try {
            DirectForwardMetalHalfLinearAdmissionPlan plan =
                    plan(true, input, weight);

            assertFalse(plan.admitted());
            assertEquals("reject:candidate_ineligible", plan.rejectionDecision());
        } finally {
            input.close();
            weight.close();
        }
    }

    @Test
    void rejectsNonHalfWeight() {
        AccelTensor input = f32(1, 3);
        AccelTensor weight = f32(2, 3);

        try {
            DirectForwardMetalHalfLinearAdmissionPlan plan =
                    plan(true, input, weight);

            assertFalse(plan.admitted());
            assertEquals("reject:candidate_ineligible", plan.rejectionDecision());
        } finally {
            input.close();
            weight.close();
        }
    }

    @Test
    void rejectsRankOneInput() {
        AccelTensor input = f32(3);
        AccelTensor weight = f16(2, 3);

        try {
            DirectForwardMetalHalfLinearAdmissionPlan plan =
                    plan(true, input, weight);

            assertFalse(plan.admitted());
            assertEquals("reject:candidate_ineligible", plan.rejectionDecision());
        } finally {
            input.close();
            weight.close();
        }
    }

    private static DirectForwardMetalHalfLinearAdmissionPlan plan(
            boolean metalLinearEnabled,
            AccelTensor input,
            AccelTensor weight) {
        return DirectForwardMetalHalfLinearAdmissionPlan.from(
                generic(),
                metalLinearEnabled,
                input,
                weight,
                PROFILE_KEY);
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
}
