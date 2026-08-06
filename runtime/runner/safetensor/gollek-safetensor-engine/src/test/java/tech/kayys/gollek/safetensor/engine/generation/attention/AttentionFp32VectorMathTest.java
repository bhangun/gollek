/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import org.junit.jupiter.api.Test;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class AttentionFp32VectorMathTest {

    @Test
    void dotsUpdatesAndWritesFp32Vectors() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment q = floats(arena, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f);
            MemorySegment key = floats(arena, 0.5f, -1.0f, 2.0f, 1.0f, -2.0f);
            MemorySegment value = floats(arena, 10.0f, 20.0f, 30.0f, 40.0f, 50.0f);
            MemorySegment out = arena.allocate(5L * Float.BYTES);
            float[] acc = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };

            float score = AttentionFp32VectorMath.dotProduct(q, 0, key, 0, 5);
            AttentionFp32VectorMath.updateAccumulator(acc, value, 0, 0.5f, 0.25f, 5);
            AttentionFp32VectorMath.writeNormalizedAccumulator(out, 0, acc, 0.5f, 5);

            assertEquals(-1.5f, score, 0.0001f);
            assertArrayEquals(new float[] { 3.0f, 6.0f, 9.0f, 12.0f, 15.0f }, acc, 0.0001f);
            assertArrayEquals(new float[] { 1.5f, 3.0f, 4.5f, 6.0f, 7.5f }, toFloats(out, 5), 0.0001f);
        }
    }

    private static MemorySegment floats(Arena arena, float... values) {
        MemorySegment segment = arena.allocate((long) values.length * Float.BYTES);
        for (int i = 0; i < values.length; i++) {
            segment.setAtIndex(ValueLayout.JAVA_FLOAT, i, values[i]);
        }
        return segment;
    }

    private static float[] toFloats(MemorySegment segment, int count) {
        float[] values = new float[count];
        for (int i = 0; i < count; i++) {
            values[i] = segment.getAtIndex(ValueLayout.JAVA_FLOAT, i);
        }
        return values;
    }
}
