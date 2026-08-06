/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import org.junit.jupiter.api.Test;
import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FlashAttentionShapeValidatorTest {

    @Test
    void reshapesProjectionWhenLastDimensionMatchesLayout() {
        AccelTensor projection = AccelTensor.zeros(1, 2, 6);

        AccelTensor reshaped = FlashAttentionShapeValidator.reshapeProjection(
                projection, "query", 1, 2, 3, 2, new ModelConfig(), 0);

        assertEquals(4, reshaped.rank());
        assertEquals(1, reshaped.size(0));
        assertEquals(2, reshaped.size(1));
        assertEquals(3, reshaped.size(2));
        assertEquals(2, reshaped.size(3));
        reshaped.close();
        projection.close();
    }

    @Test
    void failsWithActionableProjectionShapeMessage() {
        ModelConfig config = new ModelConfig();
        config.overrideNumAttentionHeads(24);
        config.overrideHeadDim(213);
        AccelTensor projection = AccelTensor.zeros(1, 9, 5120);

        IllegalArgumentException error = assertThrows(IllegalArgumentException.class,
                () -> FlashAttentionShapeValidator.validateProjection(
                        projection, "query", 24, 213, config, 0));

        String message = error.getMessage();
        assertTrue(message.contains("query projection shape mismatch"));
        assertTrue(message.contains("lastDim=5120"));
        assertTrue(message.contains("expected=5112"));
        assertTrue(message.contains("heads=24"));
        assertTrue(message.contains("headDim=213"));
        projection.close();
    }

    @Test
    void failsBeforeReshapeWithTargetShapeAndElementCounts() {
        ModelConfig config = new ModelConfig();
        config.overrideNumAttentionHeads(24);
        config.overrideHeadDim(213);
        AccelTensor projection = AccelTensor.zeros(1, 9, 5120);

        IllegalArgumentException error = assertThrows(IllegalArgumentException.class,
                () -> FlashAttentionShapeValidator.reshapeProjection(
                        projection, "query", 1, 9, 24, 213, config, 0));

        String message = error.getMessage();
        assertTrue(message.contains("query projection shape mismatch"));
        assertTrue(message.contains("lastDim=5120"));
        assertTrue(message.contains("expected=5112"));
        assertTrue(message.contains("targetShape=[1, 9, 24, 213]"));
        assertTrue(message.contains("actualElements=46080"));
        assertTrue(message.contains("expectedElements=46008"));
        projection.close();
    }

    @Test
    void failsWhenFullTargetShapeCannotPreserveProjectionElements() {
        ModelConfig config = new ModelConfig();
        AccelTensor projection = AccelTensor.zeros(1, 10, 6);

        IllegalArgumentException error = assertThrows(IllegalArgumentException.class,
                () -> FlashAttentionShapeValidator.reshapeProjection(
                        projection, "query", 1, 9, 3, 2, config, 0));

        String message = error.getMessage();
        assertTrue(message.contains("lastDim=6"));
        assertTrue(message.contains("expected=6"));
        assertTrue(message.contains("targetShape=[1, 9, 3, 2]"));
        assertTrue(message.contains("actualElements=60"));
        assertTrue(message.contains("expectedElements=54"));
        projection.close();
    }
}
