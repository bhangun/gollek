/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;
import tech.kayys.alkhawarizm.spi.model.mapper.GgufMetadataMapper;

class FlashAttentionMetalLinearWeightsTest {
    private final FlashAttentionMetalLinearWeights weights = new FlashAttentionMetalLinearWeights();
    private final FlashAttentionModelPolicy modelPolicy = FlashAttentionModelPolicy.resolve(
            null, new GgufMetadataMapper().fromGgufMetadata(Map.of("general.architecture", "llama")));

    @Test
    void acceptsF16WeightsWithoutCopying() {
        AccelTensor weight = AccelTensor.zeros(2, 2)
                .withQuantization(AccelTensor.QuantType.F16, null, null, -1);

        try {
            assertTrue(weights.canUseMixedHalfPairWeight(weight, modelPolicy));
            assertSame(weight, weights.toMetalHalfWeight(weight, false, modelPolicy));
        } finally {
            weight.close();
        }
    }

    @Test
    void rejectsF32WeightsForHalfLinearPaths() {
        AccelTensor weight = AccelTensor.zeros(2, 2);

        try {
            assertFalse(weights.canUseMixedHalfPairWeight(weight, modelPolicy));
            assertNull(weights.toMetalHalfWeight(weight, false, modelPolicy));
        } finally {
            weight.close();
        }
    }

    @Test
    void treatsMissingWeightsAsIneligible() {
        assertFalse(weights.canUseMixedHalfPairWeight(null, modelPolicy));
        assertNull(weights.toMetalHalfWeight(null, false, modelPolicy));
        assertFalse(weights.shouldUseNativeMetalBf16Linear(modelPolicy, (AccelTensor[]) null));
    }
}
