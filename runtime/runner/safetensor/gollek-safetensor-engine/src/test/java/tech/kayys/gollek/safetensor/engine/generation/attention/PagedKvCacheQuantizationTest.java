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

class PagedKvCacheQuantizationTest {

    @Test
    void int8RoundTripsVectorWithStoredScale() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment source = floats(arena, -2.0f, 0.0f, 2.0f);
            MemorySegment quantized = arena.allocate(3);
            MemorySegment scale = arena.allocate(Float.BYTES);
            MemorySegment restored = arena.allocate(3L * Float.BYTES);

            PagedKvCacheQuantization.writeVector(BlockManager.KvStorageType.INT8,
                    source, 0, quantized, 0, scale, 0, 3);
            PagedKvCacheQuantization.writeVectorAsFloat(BlockManager.KvStorageType.INT8,
                    quantized, scale, 0, 0, restored, 0, 3);

            assertArrayEquals(new float[] { -2.0f, 0.0f, 2.0f }, toFloats(restored, 3), 0.0001f);
        }
    }

    @Test
    void int4PacksSignedNibblesAndRoundTripsVectorWithStoredScale() {
        try (Arena arena = Arena.ofConfined()) {
            float oneStep = 2.0f / 7.0f;
            MemorySegment source = floats(arena, -2.0f, 0.0f, 2.0f, -oneStep);
            MemorySegment quantized = arena.allocate(2);
            MemorySegment scale = arena.allocate(Float.BYTES);
            MemorySegment restored = arena.allocate(4L * Float.BYTES);

            PagedKvCacheQuantization.writeVector(BlockManager.KvStorageType.INT4,
                    source, 0, quantized, 0, scale, 0, 4);

            assertEquals(-7, PagedKvCacheQuantization.readPackedSignedInt4(quantized, 0));
            assertEquals(0, PagedKvCacheQuantization.readPackedSignedInt4(quantized, 1));
            assertEquals(7, PagedKvCacheQuantization.readPackedSignedInt4(quantized, 2));
            assertEquals(-1, PagedKvCacheQuantization.readPackedSignedInt4(quantized, 3));

            PagedKvCacheQuantization.writeVectorAsFloat(BlockManager.KvStorageType.INT4,
                    quantized, scale, 0, 0, restored, 0, 4);
            assertArrayEquals(new float[] { -2.0f, 0.0f, 2.0f, -oneStep },
                    toFloats(restored, 4), 0.0001f);
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
