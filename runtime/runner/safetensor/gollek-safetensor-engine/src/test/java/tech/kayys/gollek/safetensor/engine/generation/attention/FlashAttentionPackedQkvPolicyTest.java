/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import org.junit.jupiter.api.Test;
import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.alkhawarizm.spi.model.ModelAttentionTraitsPolicy;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;
import tech.kayys.alkhawarizm.spi.model.ModelRuntimeTraits;

import java.lang.foreign.MemorySegment;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FlashAttentionPackedQkvPolicyTest {

    @Test
    void usesModelFamilyRuntimeTraitWhenPackedQkvIsAdvertised() {
        ModelConfig config = new ModelConfig();
        ModelRuntimeTraits traits = ModelRuntimeTraits.builder()
                .attention(ModelAttentionTraitsPolicy.phiText(config))
                .build();
        FlashAttentionModelPolicy policy = FlashAttentionModelPolicy.resolve(null, config, traits);
        AttentionInput input = inputWithWeights(config,
                AccelTensor.view(MemorySegment.NULL, new long[] { 5120, 5120 }),
                AccelTensor.view(MemorySegment.NULL, new long[] { 1024, 5120 }),
                AccelTensor.view(MemorySegment.NULL, new long[] { 1024, 5120 }));

        assertTrue(FlashAttentionPackedQkvPolicy.shouldUse(input, config, 24, 8, policy));
    }

    @Test
    void infersSharedPackedProjectionFromWeightShape() {
        ModelConfig config = new ModelConfig();
        config.overrideHeadDim(128);
        AccelTensor shared = AccelTensor.view(MemorySegment.NULL, new long[] { 5120, 3072 });
        AttentionInput input = inputWithWeights(config, shared, shared, shared);

        assertTrue(FlashAttentionPackedQkvPolicy.shouldUse(input, config, 24, 8, null));
    }

    @Test
    void doesNotInferPackedProjectionWhenSharedRowsDoNotMatchLayout() {
        ModelConfig config = new ModelConfig();
        config.overrideHeadDim(128);
        AccelTensor shared = AccelTensor.view(MemorySegment.NULL, new long[] { 5131, 3072 });
        AttentionInput input = inputWithWeights(config, shared, shared, shared);

        assertFalse(FlashAttentionPackedQkvPolicy.shouldUse(input, config, 24, 8, null));
    }

    @Test
    void resolvesPackedHeadDimFromPackedRowsWhenConfiguredDimDoesNotMatch() {
        ModelConfig config = new ModelConfig();
        config.overrideHeadDim(213);
        AccelTensor packed = AccelTensor.view(MemorySegment.NULL, new long[] { 5120, 5120 });

        assertEquals(128, FlashAttentionPackedQkvPolicy.resolveHeadDim(packed, config, 24, 8));
    }

    private static AttentionInput inputWithWeights(ModelConfig config, AccelTensor qW, AccelTensor kW,
            AccelTensor vW) {
        return new AttentionInput(
                AccelTensor.view(MemorySegment.NULL, new long[] { 1, 1, 1 }),
                qW, kW, vW, null,
                null, null, null, null,
                null, config, null, 0, 0, true,
                null, null, null, null);
    }
}
