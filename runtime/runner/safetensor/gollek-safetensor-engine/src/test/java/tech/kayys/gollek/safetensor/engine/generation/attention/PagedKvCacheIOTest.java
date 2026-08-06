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

/**
 * Verifies dense-to-paged KV cache copies only touch blocks required by the
 * current attention context.
 */
class PagedKvCacheIOTest {
    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

    @Test
    void gatherSkipsUnusedBlockTableEntries() throws Exception {
        try (KVCacheManager.KVCacheSession session = new KVCacheManager.KVCacheSession(32, new BlockManager());
                AccelTensor key = AccelTensor.fromFloatArray(new float[] { 1.0f, 2.0f }, 1, 1, 1, 2);
                AccelTensor value = AccelTensor.fromFloatArray(new float[] { 101.0f, 102.0f }, 1, 1, 1, 2);
                Arena arena = Arena.ofConfined()) {
            session.allocate(config(), GenerationConfig.defaults());
            PagedKvCacheIO.updateCache(key, value, session, 0, 0, 1, 1, 2);
            session.getBlockIndices(0).add(-1);

            MemorySegment keyOut = arena.allocate(2L * Float.BYTES, Float.BYTES);
            MemorySegment valueOut = arena.allocate(2L * Float.BYTES, Float.BYTES);
            PagedKvCacheIO.gather(session.blockManager(), session, 0, 1, 1, 2, keyOut, valueOut);

            assertArrayEquals(new float[] { 1.0f, 2.0f }, toFloats(keyOut, 2), 0.0001f);
            assertArrayEquals(new float[] { 101.0f, 102.0f }, toFloats(valueOut, 2), 0.0001f);
        }
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
