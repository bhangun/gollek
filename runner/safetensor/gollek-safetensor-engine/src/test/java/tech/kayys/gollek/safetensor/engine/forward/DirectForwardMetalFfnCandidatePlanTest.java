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

class DirectForwardMetalFfnCandidatePlanTest {
    private static final String PROFILE_KEY = "ffn_candidate_test";

    @Test
    void fusedPlanAdmitsValidHalfCandidates() {
        try (TestTensors tensors = TestTensors.valid()) {
            DirectForwardMetalFfnCandidatePlan plan = DirectForwardMetalFfnCandidatePlan.fused(
                    true,
                    tensors.input,
                    tensors.gateW,
                    tensors.upW,
                    tensors.downW,
                    generic(),
                    PROFILE_KEY);

            assertTrue(plan.admitted());
            assertNull(plan.rejectionReason());
            assertNull(plan.rejectionDecision());
        }
    }

    @Test
    void fusedPlanReportsGateBeforeUpAndDown() {
        try (TestTensors tensors = TestTensors.valid()) {
            DirectForwardMetalFfnCandidatePlan plan = DirectForwardMetalFfnCandidatePlan.fused(
                    false,
                    tensors.input,
                    tensors.gateW,
                    null,
                    null,
                    generic(),
                    PROFILE_KEY);

            assertFalse(plan.admitted());
            assertEquals("gate_candidate_ineligible", plan.rejectionReason());
            assertEquals("reject:gate_candidate_ineligible", plan.rejectionDecision());
        }
    }

    @Test
    void fusedPlanReportsUpBeforeDown() {
        try (TestTensors tensors = TestTensors.valid()) {
            DirectForwardMetalFfnCandidatePlan plan = DirectForwardMetalFfnCandidatePlan.fused(
                    true,
                    tensors.input,
                    tensors.gateW,
                    null,
                    null,
                    generic(),
                    PROFILE_KEY);

            assertFalse(plan.admitted());
            assertEquals("up_candidate_ineligible", plan.rejectionReason());
        }
    }

    @Test
    void fusedPlanReportsDownAfterGateAndUpPass() {
        try (TestTensors tensors = TestTensors.valid()) {
            DirectForwardMetalFfnCandidatePlan plan = DirectForwardMetalFfnCandidatePlan.fused(
                    true,
                    tensors.input,
                    tensors.gateW,
                    tensors.upW,
                    null,
                    generic(),
                    PROFILE_KEY);

            assertFalse(plan.admitted());
            assertEquals("down_weight_ineligible", plan.rejectionReason());
        }
    }

    @Test
    void matvecGatedPlanCollapsesCandidateFailures() {
        try (TestTensors tensors = TestTensors.valid()) {
            DirectForwardMetalFfnCandidatePlan plan = DirectForwardMetalFfnCandidatePlan.matvecGated(
                    true,
                    tensors.input,
                    tensors.gateW,
                    tensors.upW,
                    null,
                    generic(),
                    PROFILE_KEY);

            assertFalse(plan.admitted());
            assertEquals("candidate_ineligible", plan.rejectionReason());
            assertEquals("reject:candidate_ineligible", plan.rejectionDecision());
        }
    }

    @Test
    void gateUpMatvecPlanCollapsesCandidateFailures() {
        try (TestTensors tensors = TestTensors.valid()) {
            DirectForwardMetalFfnCandidatePlan plan = DirectForwardMetalFfnCandidatePlan.gateUpMatvec(
                    true,
                    tensors.input,
                    tensors.gateW,
                    null,
                    generic(),
                    PROFILE_KEY);

            assertFalse(plan.admitted());
            assertEquals("candidate_ineligible", plan.rejectionReason());
            assertEquals("reject:candidate_ineligible", plan.rejectionDecision());
        }
    }

    private static AccelTensor f32(long... shape) {
        return AccelTensor.zeros(shape);
    }

    private static AccelTensor f16(long... shape) {
        return AccelTensor.zeros(shape)
                .withQuantization(AccelTensor.QuantType.F16, null, null, -1);
    }

    private static ModelConfigTraits generic() {
        return new ModelConfigTraits(null, "llama", 0, 0, false, false, false, false, false, false, false);
    }

    private static final class TestTensors implements AutoCloseable {
        private final AccelTensor input;
        private final AccelTensor gateW;
        private final AccelTensor upW;
        private final AccelTensor downW;

        private TestTensors(AccelTensor input, AccelTensor gateW, AccelTensor upW, AccelTensor downW) {
            this.input = input;
            this.gateW = gateW;
            this.upW = upW;
            this.downW = downW;
        }

        private static TestTensors valid() {
            return new TestTensors(f32(1, 3), f16(2, 3), f16(2, 3), f16(3, 2));
        }

        @Override
        public void close() {
            input.close();
            gateW.close();
            upW.close();
            downW.close();
        }
    }
}
