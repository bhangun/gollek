/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import org.junit.jupiter.api.Test;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;
import java.util.BitSet;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class GenerationGreedyArgmaxTest {

    @Test
    void selectJavaArrayReturnsHighestFiniteLogit() {
        float[] logits = { 0.25f, Float.NaN, 1.5f, -2.0f };

        assertEquals(2, GenerationGreedyArgmax.selectJava(logits));
    }

    @Test
    void selectJavaArraySkipsRejectedCandidates() {
        float[] logits = { 0.25f, 4.0f, 3.5f, 1.0f };

        assertEquals(2, GenerationGreedyArgmax.selectJava(logits, logits.length, new int[] { 1 }, 1));
    }

    @Test
    void selectJavaArrayIgnoresUnusedRejectedCandidateBuffer() {
        float[] logits = { 0.25f, 4.0f, 3.5f, 1.0f };

        assertEquals(1, GenerationGreedyArgmax.selectJava(logits, logits.length, new int[] { 1 }, 0));
    }

    @Test
    void selectJavaArrayReturnsMinusOneWhenEveryCandidateIsRejectedOrNan() {
        float[] logits = { Float.NaN, 4.0f, 3.5f };

        assertEquals(-1, GenerationGreedyArgmax.selectJava(logits, logits.length, new int[] { 1, 2 }, 2));
    }

    @Test
    void selectJavaReturnsHighestFiniteLogit() {
        try (Arena arena = Arena.ofShared()) {
            MemorySegment logits = logits(arena, 0.25f, Float.NaN, 1.5f, -2.0f);

            assertEquals(2, GenerationGreedyArgmax.selectJava(logits, 4, null, 0));
        }
    }

    @Test
    void selectJavaSkipsRejectedCandidates() {
        try (Arena arena = Arena.ofShared()) {
            MemorySegment logits = logits(arena, 0.25f, 4.0f, 3.5f, 1.0f);

            assertEquals(2, GenerationGreedyArgmax.selectJava(logits, 4, new int[] { 1 }, 1));
        }
    }

    @Test
    void selectJavaIgnoresUnusedRejectedCandidateBuffer() {
        try (Arena arena = Arena.ofShared()) {
            MemorySegment logits = logits(arena, 0.25f, 4.0f, 3.5f, 1.0f);

            assertEquals(1, GenerationGreedyArgmax.selectJava(logits, 4, new int[] { 1 }, 0));
        }
    }

    @Test
    void selectJavaSkipsMaskAndRejectedCandidates() {
        try (Arena arena = Arena.ofShared()) {
            MemorySegment logits = logits(arena, 0.25f, 2.0f, 4.0f, 3.5f);
            BitSet mask = new BitSet();
            mask.set(2);

            assertEquals(1, GenerationGreedyArgmax.selectJava(logits, 4, mask, new int[] { 3 }, 1));
        }
    }

    @Test
    void selectJavaReturnsMinusOneWhenEveryCandidateIsRejectedOrNan() {
        try (Arena arena = Arena.ofShared()) {
            MemorySegment logits = logits(arena, Float.NaN, 4.0f, 3.5f);

            assertEquals(-1, GenerationGreedyArgmax.selectJava(logits, 3, new int[] { 1, 2 }, 2));
        }
    }

    @Test
    void selectJavaReturnsMinusOneForUnsupportedLongVocab() {
        try (Arena arena = Arena.ofShared()) {
            MemorySegment logits = logits(arena, 1.0f);

            assertEquals(-1, GenerationGreedyArgmax.selectJava(logits, (long) Integer.MAX_VALUE + 1, null, 0));
        }
    }

    @Test
    void selectWithMaskRecordsMaskAwareArgmaxPath() {
        String previousProfile = System.getProperty("gollek.profile");
        System.setProperty("gollek.profile", "true");
        GenerationGreedyArgmax.setNativeArgmaxDisabledForTest(true);
        try (Arena arena = Arena.ofShared()) {
            InferenceProfile profile = DirectInferenceProfiler.startProfile("test");
            MemorySegment logits = logits(arena, 0.25f, 2.0f, 4.0f);
            BitSet mask = new BitSet();
            mask.set(2);

            assertEquals(1, GenerationGreedyArgmax.selectWithMask(logits, 3, mask, null, 0));
            assertTrue(profile.summary("cpu").contains("argmax_paths={java_memory_segment_mask=1}"));
            Map<String, Object> metadata = profile.metadata("cpu");
            assertEquals(1, metadata.get("profile_argmax_path_java_memory_segment_mask_count"));
        } finally {
            DirectInferenceProfiler.clearProfile();
            restoreProperty("gollek.profile", previousProfile);
            GenerationGreedyArgmax.setNativeArgmaxDisabledForTest(null);
        }
    }

    @Test
    void compactNativeRejectionsDeduplicatesAndIgnoresOutOfRangeTokens() {
        int[] compact = new int[8];
        BitSet mask = new BitSet();
        mask.set(2);
        mask.set(4);
        mask.set(9);

        int count = GenerationGreedyArgmax.compactNativeRejectionsForTest(
                5, mask, new int[] { 1, 2, -1, 7 }, 4, compact);

        assertEquals(3, count);
        assertArrayEquals(new int[] { 1, 2, 4 }, Arrays.copyOf(compact, count));
    }

    @Test
    void compactNativeRejectionsRejectsMoreThanNativeLimit() {
        int[] compact = new int[8];
        BitSet mask = new BitSet();
        mask.set(0, 9);

        assertEquals(-1, GenerationGreedyArgmax.compactNativeRejectionsForTest(16, mask, null, 0, compact));
    }

    @Test
    void selectRecordsJavaMemorySegmentArgmaxPathWhenNativeDisabled() {
        String previousProfile = System.getProperty("gollek.profile");
        System.setProperty("gollek.profile", "true");
        GenerationGreedyArgmax.setNativeArgmaxDisabledForTest(true);
        try (Arena arena = Arena.ofShared()) {
            InferenceProfile profile = DirectInferenceProfiler.startProfile("test");
            MemorySegment logits = logits(arena, 0.25f, 1.0f, 3.5f);

            assertEquals(2, GenerationGreedyArgmax.select(logits, 3, null, 0));
            assertTrue(profile.summary("cpu").contains("argmax_paths={java_memory_segment=1}"));
            Map<String, Object> metadata = profile.metadata("cpu");
            assertEquals(1, metadata.get("profile_argmax_path_java_memory_segment_count"));
        } finally {
            DirectInferenceProfiler.clearProfile();
            restoreProperty("gollek.profile", previousProfile);
            GenerationGreedyArgmax.setNativeArgmaxDisabledForTest(null);
        }
    }

    private static MemorySegment logits(Arena arena, float... values) {
        MemorySegment segment = arena.allocate(ValueLayout.JAVA_FLOAT.byteSize() * values.length);
        for (int i = 0; i < values.length; i++) {
            segment.setAtIndex(ValueLayout.JAVA_FLOAT, i, values[i]);
        }
        return segment;
    }

    private static void restoreProperty(String key, String value) {
        if (value == null) {
            System.clearProperty(key);
        } else {
            System.setProperty(key, value);
        }
    }
}
