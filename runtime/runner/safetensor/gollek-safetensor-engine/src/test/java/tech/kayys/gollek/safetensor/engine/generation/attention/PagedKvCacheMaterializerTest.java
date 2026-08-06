/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.kv.BlockManager;
import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;
import tech.kayys.gollek.safetensor.generation.GenerationConfig;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

class PagedKvCacheMaterializerTest {
    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

    @Test
    void packsDenseSharedKvIntoHeadMajorPagedPool() {
        try (AccelTensor key = tensor(1.0f);
                AccelTensor value = tensor(101.0f);
                Arena arena = Arena.ofConfined()) {
            MemorySegment packedKey = arena.allocate(8L * Float.BYTES, 64);
            MemorySegment packedValue = arena.allocate(8L * Float.BYTES, 64);

            PagedKvCacheMaterializer.packDenseSharedKvIntoTemporaryPagedPool(
                    key, value, 2, 2, packedKey, packedValue);

            assertArrayEquals(new float[] { 1.0f, 2.0f, 5.0f, 6.0f, 3.0f, 4.0f, 7.0f, 8.0f },
                    toFloats(packedKey, 8), 0.0001f);
            assertArrayEquals(new float[] { 101.0f, 102.0f, 105.0f, 106.0f, 103.0f, 104.0f, 107.0f, 108.0f },
                    toFloats(packedValue, 8), 0.0001f);
        }
    }

    @Test
    void packsRangeAcrossSourceBlockBoundary() throws Exception {
        try (KVCacheManager.KVCacheSession session = new KVCacheManager.KVCacheSession(32, new BlockManager());
                AccelTensor key = tokenTensor(17, 0.0f);
                AccelTensor value = tokenTensor(17, 1000.0f);
                Arena arena = Arena.ofConfined()) {
            session.allocate(config(), GenerationConfig.defaults());
            PagedKvCacheIO.updateCache(key, value, session, 0, 0, 17, 1, 2);

            MemorySegment packedKey = arena.allocate(32L * Float.BYTES, 64);
            MemorySegment packedValue = arena.allocate(32L * Float.BYTES, 64);
            PagedKvCacheMaterializer.packRangeIntoTemporaryPagedPool(session.blockManager(), session, 0,
                    15, 17, 1, 2, 16, packedKey, packedValue);

            assertArrayEquals(new float[] { 151.0f, 152.0f, 161.0f, 162.0f },
                    toFloats(packedKey, 4), 0.0001f);
            assertArrayEquals(new float[] { 1151.0f, 1152.0f, 1161.0f, 1162.0f },
                    toFloats(packedValue, 4), 0.0001f);
        }
    }

    private static AccelTensor tensor(float start) {
        AccelTensor tensor = AccelTensor.zeros(1, 2, 2, 2);
        for (int i = 0; i < 8; i++) {
            tensor.setFlat(i, start + i);
        }
        return tensor;
    }

    private static AccelTensor tokenTensor(int tokens, float base) {
        float[] values = new float[tokens * 2];
        for (int tok = 0; tok < tokens; tok++) {
            values[tok * 2] = base + tok * 10.0f + 1.0f;
            values[tok * 2 + 1] = base + tok * 10.0f + 2.0f;
        }
        return AccelTensor.fromFloatArray(values, 1, tokens, 1, 2);
    }

    private static ModelConfig config() throws Exception {
        return OBJECT_MAPPER.readValue("""
                {
                  "model_type": "gemma4_text",
                  "hidden_size": 2,
                  "intermediate_size": 8,
                  "num_hidden_layers": 1,
                  "num_attention_heads": 1,
                  "num_key_value_heads": 1,
                  "head_dim": 2
                }
                """, ModelConfig.class);
    }

    private static float[] toFloats(MemorySegment segment, int count) {
        float[] values = new float[count];
        for (int i = 0; i < count; i++) {
            values[i] = segment.getAtIndex(ValueLayout.JAVA_FLOAT, i);
        }
        return values;
    }
}
