/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.safetensor.engine.generation.kv.BlockManager;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class PagedAttentionVectorMathTest {

    @Test
    void scoresUpdatesAndWritesFp32Vectors() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment q = floats(arena, 1.0f, 2.0f, 3.0f, 4.0f);
            MemorySegment key = floats(arena, 0.5f, -1.0f, 2.0f, 1.0f);
            MemorySegment value = floats(arena, 10.0f, 20.0f, 30.0f, 40.0f);
            MemorySegment out = arena.allocate(4L * Float.BYTES);
            float[] acc = { 1.0f, 2.0f, 3.0f, 4.0f };

            float score = PagedAttentionVectorMath.score(BlockManager.KvStorageType.FP32,
                    q, 0, key, 0, 0, null, 0, 4, 0.5f);
            PagedAttentionVectorMath.updateAccumulator(BlockManager.KvStorageType.FP32,
                    acc, value, 0, 0, null, 0, 0.5f, 0.25f, 4);
            PagedAttentionVectorMath.writeNormalizedAccumulator(out, 0, acc, 0.5f, 4);

            assertEquals(4.25f, score, 0.0001f);
            assertArrayEquals(new float[] { 3.0f, 6.0f, 9.0f, 12.0f }, acc, 0.0001f);
            assertArrayEquals(new float[] { 1.5f, 3.0f, 4.5f, 6.0f }, toFloats(out, 4), 0.0001f);
        }
    }

    @Test
    void scoresAndUpdatesInt8VectorsWithStoredScale() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment q = floats(arena, 1.0f, 2.0f, 3.0f);
            MemorySegment key = bytes(arena, 2, -4, 6);
            MemorySegment value = bytes(arena, 4, 6, -8);
            MemorySegment keyScale = floats(arena, 0.5f);
            MemorySegment valueScale = floats(arena, 0.25f);
            float[] acc = { 2.0f, 4.0f, 6.0f };

            float score = PagedAttentionVectorMath.score(BlockManager.KvStorageType.INT8,
                    q, 0, key, 0, 0, keyScale, 0, 3, 2.0f);
            PagedAttentionVectorMath.updateAccumulator(BlockManager.KvStorageType.INT8,
                    acc, value, 0, 0, valueScale, 0, 0.5f, 2.0f, 3);

            assertEquals(12.0f, score, 0.0001f);
            assertArrayEquals(new float[] { 3.0f, 5.0f, -1.0f }, acc, 0.0001f);
        }
    }

    @Test
    void scoresAndUpdatesInt4VectorsWithStoredScale() {
        try (Arena arena = Arena.ofConfined()) {
            float oneStep = 2.0f / 7.0f;
            MemorySegment q = floats(arena, 1.0f, 1.0f, 1.0f, 1.0f);
            MemorySegment keySource = floats(arena, -2.0f, 0.0f, 2.0f, -oneStep);
            MemorySegment valueSource = floats(arena, 2.0f, -2.0f, 0.0f, oneStep);
            MemorySegment key = arena.allocate(2);
            MemorySegment value = arena.allocate(2);
            MemorySegment keyScale = arena.allocate(Float.BYTES);
            MemorySegment valueScale = arena.allocate(Float.BYTES);
            float[] acc = { 0.0f, 0.0f, 0.0f, 0.0f };

            PagedKvCacheQuantization.writeVector(BlockManager.KvStorageType.INT4,
                    keySource, 0, key, 0, keyScale, 0, 4);
            PagedKvCacheQuantization.writeVector(BlockManager.KvStorageType.INT4,
                    valueSource, 0, value, 0, valueScale, 0, 4);

            float score = PagedAttentionVectorMath.score(BlockManager.KvStorageType.INT4,
                    q, 0, key, 0, 0, keyScale, 0, 4, 3.0f);
            PagedAttentionVectorMath.updateAccumulator(BlockManager.KvStorageType.INT4,
                    acc, value, 0, 0, valueScale, 0, 0.0f, 1.0f, 4);

            assertEquals(-3.0f * oneStep, score, 0.0001f);
            assertArrayEquals(new float[] { 2.0f, -2.0f, 0.0f, oneStep }, acc, 0.0001f);
        }
    }

    private static MemorySegment floats(Arena arena, float... values) {
        MemorySegment segment = arena.allocate((long) values.length * Float.BYTES);
        for (int i = 0; i < values.length; i++) {
            segment.setAtIndex(ValueLayout.JAVA_FLOAT, i, values[i]);
        }
        return segment;
    }

    private static MemorySegment bytes(Arena arena, int... values) {
        MemorySegment segment = arena.allocate(values.length);
        for (int i = 0; i < values.length; i++) {
            segment.setAtIndex(ValueLayout.JAVA_BYTE, i, (byte) values[i]);
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
