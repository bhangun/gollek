/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import org.junit.jupiter.api.Test;
import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;

import java.lang.foreign.MemorySegment;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FlashAttentionShapeAdmissionPlanTest {

    @Test
    void separateQueryWeightAdmissionDerivesHeadDimFromRows() {
        AccelTensor queryWeight = AccelTensor.view(MemorySegment.NULL, new long[] {3072, 3072});

        FlashAttentionShapeAdmissionPlan plan =
                FlashAttentionShapeAdmissionPlan.separateQueryWeightLayout(
                        queryWeight, new ModelConfig(), 24, 0);

        assertTrue(plan.admitted());
        assertEquals(128, plan.getResolvedHeadDim());
        assertNull(plan.rejectionMessage());
    }

    @Test
    void separateQueryWeightAdmissionRejectsNonDivisibleRowsWithModelFamilyHint() {
        ModelConfig config = new ModelConfig();
        config.overrideHiddenSize(5120);
        config.overrideHeadDim(213);
        AccelTensor queryWeight = AccelTensor.view(MemorySegment.NULL, new long[] {5120, 5120});

        FlashAttentionShapeAdmissionPlan plan =
                FlashAttentionShapeAdmissionPlan.separateQueryWeightLayout(queryWeight, config, 24, 0);

        assertFalse(plan.admitted());
        assertTrue(plan.rejectionMessage().contains("query projection layout mismatch"));
        assertTrue(plan.rejectionMessage().contains("rows=5120"));
        assertTrue(plan.rejectionMessage().contains("numQueryHeads=24"));
        assertTrue(plan.rejectionMessage().contains("configuredHeadDim=213"));
        assertTrue(plan.rejectionMessage().contains("model-family runtime traits"));
    }

    @Test
    void packedQkvWeightAdmissionRejectsRowsThatDoNotMatchLayout() {
        AccelTensor packedWeight = AccelTensor.view(MemorySegment.NULL, new long[] {5112, 3072});
        FlashAttentionHeadLayout layout = new FlashAttentionHeadLayout(24, 8, 128, true);

        FlashAttentionShapeAdmissionPlan plan =
                FlashAttentionShapeAdmissionPlan.packedQkvWeight(packedWeight, layout);

        assertFalse(plan.admitted());
        assertTrue(plan.rejectionMessage().contains("rows=5112"));
        assertTrue(plan.rejectionMessage().contains("expected=5120"));
        assertTrue(plan.rejectionMessage().contains("qHeads=24"));
        assertTrue(plan.rejectionMessage().contains("kvHeads=8"));
        assertTrue(plan.rejectionMessage().contains("headDim=128"));
    }

    @Test
    void packedQkvOutputAdmissionRejectsMissingProjection() {
        FlashAttentionHeadLayout layout = new FlashAttentionHeadLayout(24, 8, 128, true);

        FlashAttentionShapeAdmissionPlan plan =
                FlashAttentionShapeAdmissionPlan.packedQkvOutput(null, layout);

        assertFalse(plan.admitted());
        assertTrue(plan.rejectionMessage().contains("lastDim=<missing>"));
        assertTrue(plan.rejectionMessage().contains("expected=5120"));
    }

    @Test
    void packedQkvOutputAdmissionAcceptsMatchingProjectionLastDim() {
        AccelTensor packedProjection = AccelTensor.view(MemorySegment.NULL, new long[] {1, 9, 5120});
        FlashAttentionHeadLayout layout = new FlashAttentionHeadLayout(24, 8, 128, true);

        FlashAttentionShapeAdmissionPlan plan =
                FlashAttentionShapeAdmissionPlan.packedQkvOutput(packedProjection, layout);

        assertTrue(plan.admitted());
        assertNull(plan.rejectionMessage());
    }
}
