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
import static org.junit.jupiter.api.Assertions.assertEquals;

class FlashAttentionJavaFallbackTest {
    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

    @Test
    void denseSharedAttentionMixesValuesWithOnlineSoftmax() {
        try (AccelTensor query = AccelTensor.fromFloatArray(new float[] { 1.0f, 0.0f }, 1, 1, 1, 2);
                AccelTensor key = AccelTensor.fromFloatArray(new float[] {
                        1.0f, 0.0f,
                        0.0f, 1.0f
                }, 1, 2, 1, 2);
                AccelTensor value = AccelTensor.fromFloatArray(new float[] {
                        10.0f, 0.0f,
                        0.0f, 20.0f
                }, 1, 2, 1, 2);
                AccelTensor output = FlashAttentionJavaFallback.denseSharedAttention(
                        query, key, value, null, 0, 0, 1, 1, 2, 1.0f, false, 0.0f)) {
            float expOne = (float) Math.exp(1.0f);
            float invSum = 1.0f / (expOne + 1.0f);

            assertArrayEquals(new float[] { 10.0f * expOne * invSum, 20.0f * invSum },
                    output.toFloatArray(), 0.0001f);
        }
    }

    @Test
    void denseSharedAttentionMasksFutureKeysWhenCausal() {
        try (AccelTensor query = AccelTensor.fromFloatArray(new float[] {
                        1.0f, 0.0f,
                        0.0f, 1.0f
                }, 1, 2, 1, 2);
                AccelTensor key = AccelTensor.fromFloatArray(new float[] {
                        1.0f, 0.0f,
                        0.0f, 1.0f
                }, 1, 2, 1, 2);
                AccelTensor value = AccelTensor.fromFloatArray(new float[] {
                        10.0f, 0.0f,
                        0.0f, 20.0f
                }, 1, 2, 1, 2);
                AccelTensor output = FlashAttentionJavaFallback.denseSharedAttention(
                        query, key, value, null, 0, 0, 1, 1, 2, 1.0f, true, 0.0f)) {
            float expOne = (float) Math.exp(1.0f);
            float invSum = 1.0f / (expOne + 1.0f);

            assertArrayEquals(new float[] {
                    10.0f, 0.0f,
                    10.0f * invSum, 20.0f * expOne * invSum
            }, output.toFloatArray(), 0.0001f);
        }
    }

    @Test
    void denseSharedAttentionWritesIntoReusableContextBuffer() {
        try (Arena arena = Arena.ofConfined();
                AccelTensor query = AccelTensor.fromFloatArray(new float[] { 1.0f, 0.0f }, 1, 1, 1, 2);
                AccelTensor key = AccelTensor.fromFloatArray(new float[] { 1.0f, 0.0f }, 1, 1, 1, 2);
                AccelTensor value = AccelTensor.fromFloatArray(new float[] { 3.0f, 4.0f }, 1, 1, 1, 2)) {
            MemorySegment contextBuffer = arena.allocate(2L * Float.BYTES, Float.BYTES);
            try (AccelTensor output = FlashAttentionJavaFallback.denseSharedAttention(
                    query, key, value, null, 0, 0, 1, 1, 2, 1.0f, false, 0.0f, contextBuffer)) {
                assertArrayEquals(new float[] { 3.0f, 4.0f }, output.toFloatArray(), 0.0001f);

                output.setFlat(0, 6.25f);
                assertEquals(6.25f, contextBuffer.get(ValueLayout.JAVA_FLOAT, 0), 0.0001f);
            }
        }
    }

    @Test
    void denseSharedAttentionIgnoresContextBufferThatAliasesQuery() {
        try (AccelTensor query = AccelTensor.fromFloatArray(new float[] { 1.0f, 0.0f }, 1, 1, 1, 2);
                AccelTensor key = AccelTensor.fromFloatArray(new float[] { 1.0f, 0.0f }, 1, 1, 1, 2);
                AccelTensor value = AccelTensor.fromFloatArray(new float[] { 3.0f, 4.0f }, 1, 1, 1, 2);
                AccelTensor output = FlashAttentionJavaFallback.denseSharedAttention(
                        query, key, value, null, 0, 0, 1, 1, 2, 1.0f, false, 0.0f, query.dataPtr())) {
            assertArrayEquals(new float[] { 3.0f, 4.0f }, output.toFloatArray(), 0.0001f);

            output.setFlat(0, 9.5f);
            assertEquals(1.0f, query.getFlat(0), 0.0001f);
        }
    }

    @Test
    void denseCachedAttentionGathersKvIntoReusableWorkspaceScratch() throws Exception {
        ModelConfig config = denseCacheConfig();
        try (KVCacheManager.KVCacheSession session = new KVCacheManager.KVCacheSession(16, new BlockManager());
                AccelTensor query = AccelTensor.fromFloatArray(new float[] { 1.0f, 0.0f }, 1, 1, 1, 2);
                AccelTensor key = AccelTensor.fromFloatArray(new float[] { 1.0f, 0.0f }, 1, 1, 1, 2);
                AccelTensor value = AccelTensor.fromFloatArray(new float[] { 3.0f, 4.0f }, 1, 1, 1, 2)) {
            session.allocate(config, GenerationConfig.defaults());
            PagedKvCacheIO.updateCache(key, value, session, 0, 0, 1, 1, 2);

            try (AccelTensor output = FlashAttentionJavaFallback.denseCachedAttention(
                    query, session, 0, 0, 1, 1, 2, 1.0f, false, 0.0f, config, 0)) {
                assertArrayEquals(new float[] { 3.0f, 4.0f }, output.toFloatArray(), 0.0001f);
                assertEquals(1.0f, session.getWorkspace().getGateSeg().getAtIndex(ValueLayout.JAVA_FLOAT, 0),
                        0.0001f);
                assertEquals(3.0f, session.getWorkspace().getUpSeg().getAtIndex(ValueLayout.JAVA_FLOAT, 0),
                        0.0001f);
            }
        }
    }

    private static ModelConfig denseCacheConfig() throws Exception {
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
}
