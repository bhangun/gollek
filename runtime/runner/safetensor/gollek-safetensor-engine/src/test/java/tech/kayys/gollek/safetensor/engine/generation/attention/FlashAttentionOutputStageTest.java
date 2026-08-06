/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import org.junit.jupiter.api.Test;
import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Verifies attention output reshaping and O projection workspace reuse.
 */
class FlashAttentionOutputStageTest {
    @Test
    void reusesAttentionOutputBufferForOutputProjectionFallback() {
        try (Arena arena = Arena.ofConfined();
                AccelTensor x = AccelTensor.zeros(1, 1, 8);
                AccelTensor attentionOutput = attentionOutput();
                AccelTensor outputWeight = identityWeight(8)) {
            MemorySegment outputBuffer = arena.allocate(8L * Float.BYTES, Float.BYTES);
            AccelTensor projected = outputStage().project(
                    input(x, outputWeight, outputBuffer),
                    attentionOutput,
                    1,
                    new FlashAttentionHeadLayout(2, 2, 4, false),
                    new ModelConfig(),
                    null,
                    false);

            try {
                assertArrayEquals(new float[] {
                        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f
                }, projected.toFloatArray(), 0.0001f);

                projected.setFlat(0, 42.0f);
                assertEquals(42.0f, outputBuffer.get(ValueLayout.JAVA_FLOAT, 0), 0.0001f);
            } finally {
                projected.close();
            }
        }
    }

    @Test
    void reusesAttentionOutputBufferForPostAttentionRmsNormFallback() {
        try (Arena arena = Arena.ofConfined();
                AccelTensor x = AccelTensor.zeros(1, 1, 8);
                AccelTensor attentionOutput = attentionOutput();
                AccelTensor outputWeight = identityWeight(8);
                AccelTensor postAttentionNorm = AccelTensor.ones(8)) {
            MemorySegment outputBuffer = arena.allocate(8L * Float.BYTES, Float.BYTES);
            AccelTensor projected = outputStage().project(
                    input(x, outputWeight, postAttentionNorm, outputBuffer),
                    attentionOutput,
                    1,
                    new FlashAttentionHeadLayout(2, 2, 4, false),
                    new ModelConfig(),
                    null,
                    false);

            try {
                projected.setFlat(0, 24.0f);
                assertEquals(24.0f, outputBuffer.get(ValueLayout.JAVA_FLOAT, 0), 0.0001f);
            } finally {
                projected.close();
            }
        }
    }

    @Test
    void ignoresUndersizedAttentionOutputBufferForOutputProjectionFallback() {
        try (Arena arena = Arena.ofConfined();
                AccelTensor x = AccelTensor.zeros(1, 1, 8);
                AccelTensor attentionOutput = attentionOutput();
                AccelTensor outputWeight = identityWeight(8)) {
            MemorySegment tooSmallOutputBuffer = arena.allocate(4L, Float.BYTES);
            tooSmallOutputBuffer.set(ValueLayout.JAVA_FLOAT, 0, 0.0f);
            AccelTensor projected = outputStage().project(
                    input(x, outputWeight, tooSmallOutputBuffer),
                    attentionOutput,
                    1,
                    new FlashAttentionHeadLayout(2, 2, 4, false),
                    new ModelConfig(),
                    null,
                    false);

            try {
                projected.setFlat(0, 42.0f);
                assertEquals(0.0f, tooSmallOutputBuffer.get(ValueLayout.JAVA_FLOAT, 0), 0.0001f);
            } finally {
                projected.close();
            }
        }
    }

    private static FlashAttentionOutputStage outputStage() {
        return new FlashAttentionOutputStage(
                new FlashAttentionProjector(null, () -> false),
                new FlashAttentionNormalizer(() -> null));
    }

    private static AttentionInput input(AccelTensor x, AccelTensor outputWeight, MemorySegment outputBuffer) {
        return input(x, outputWeight, null, outputBuffer);
    }

    private static AttentionInput input(AccelTensor x, AccelTensor outputWeight, AccelTensor postAttentionNorm,
            MemorySegment outputBuffer) {
        return new AttentionInput(
                x, null, null, null, outputWeight,
                null, null, null, null,
                null, new ModelConfig(), null, 0, 0, true,
                null, null, postAttentionNorm, null,
                outputBuffer);
    }

    private static AccelTensor attentionOutput() {
        return AccelTensor.fromFloatArray(new float[] {
                1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f
        }, 1, 1, 2, 4);
    }

    private static AccelTensor identityWeight(int dim) {
        float[] weights = new float[dim * dim];
        for (int i = 0; i < dim; i++) {
            weights[i * dim + i] = 1.0f;
        }
        return AccelTensor.fromFloatArray(weights, dim, dim);
    }
}
