/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class FlashAttentionDenseFallbackLoopTest {

    @Test
    void gatheredSourceSupportsGqaHeadMapping() {
        try (Arena arena = Arena.ofConfined();
                AccelTensor query = AccelTensor.fromFloatArray(new float[] {
                        1.0f, 0.0f,
                        0.0f, 1.0f
                }, 1, 1, 2, 2);
                AccelTensor output = FlashAttentionDenseFallbackLoop.compute(
                        query,
                        new GatheredSource(
                                floats(arena,
                                        1.0f, 0.0f,
                                        0.0f, 1.0f),
                                floats(arena,
                                        10.0f, 0.0f,
                                        0.0f, 20.0f),
                                2,
                                1,
                                2),
                        null, 0, 0, 2, 1, 2, 1.0f, false, 0.0f)) {
            float expOne = (float) Math.exp(1.0f);
            float invSum = 1.0f / (expOne + 1.0f);

            assertArrayEquals(new float[] {
                    10.0f * expOne * invSum, 20.0f * invSum,
                    10.0f * invSum, 20.0f * expOne * invSum
            }, output.toFloatArray(), 0.0001f);
        }
    }

    @Test
    void slidingWindowSkipsTokensBeforeWindow() throws Exception {
        try (Arena arena = Arena.ofConfined();
                AccelTensor query = AccelTensor.fromFloatArray(new float[] { 1.0f, 0.0f }, 1, 1, 1, 2);
                AccelTensor output = FlashAttentionDenseFallbackLoop.compute(
                        query,
                        new GatheredSource(
                                floats(arena,
                                        -20.0f, 0.0f,
                                        1.0f, 0.0f,
                                        0.0f, 1.0f),
                                floats(arena,
                                        1000.0f, 1000.0f,
                                        10.0f, 0.0f,
                                        0.0f, 20.0f),
                                3,
                                1,
                                2),
                        slidingConfig(), 0, 2, 1, 1, 2, 1.0f, false, 0.0f)) {
            float expOne = (float) Math.exp(1.0f);
            float invSum = 1.0f / (expOne + 1.0f);

            assertArrayEquals(new float[] { 10.0f * expOne * invSum, 20.0f * invSum },
                    output.toFloatArray(), 0.0001f);
        }
    }

    private record GatheredSource(
            MemorySegment keySegment,
            MemorySegment valueSegment,
            int totalTokens,
            int numKVHeads,
            int headDim) implements FlashAttentionDenseFallbackLoop.KeyValueSource {

        @Override
        public long keyOffset(int batch, int token, int kvHeadIdx) {
            return ((long) token * numKVHeads + kvHeadIdx) * headDim;
        }

        @Override
        public long valueOffset(int batch, int token, int kvHeadIdx) {
            return ((long) token * numKVHeads + kvHeadIdx) * headDim;
        }
    }

    private static MemorySegment floats(Arena arena, float... values) {
        MemorySegment segment = arena.allocate((long) values.length * Float.BYTES);
        for (int i = 0; i < values.length; i++) {
            segment.setAtIndex(ValueLayout.JAVA_FLOAT, i, values[i]);
        }
        return segment;
    }

    private static ModelConfig slidingConfig() throws Exception {
        return new ObjectMapper().readValue("""
                {
                  "model_type": "test",
                  "sliding_window": 2,
                  "layer_types": ["sliding_attention"]
                }
                """, ModelConfig.class);
    }
}
