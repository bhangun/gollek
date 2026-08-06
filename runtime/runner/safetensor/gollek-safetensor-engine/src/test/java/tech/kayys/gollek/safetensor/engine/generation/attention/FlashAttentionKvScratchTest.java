/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.safetensor.engine.generation.kv.BlockManager;
import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;
import tech.kayys.gollek.safetensor.generation.GenerationConfig;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

/**
 * Verifies attention KV materialization uses reusable forward workspace scratch
 * instead of per-dispatch temporary arenas.
 */
class FlashAttentionKvScratchTest {
    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

    @Test
    void projectionScratchPoolsUseProjectionScratchSegments() throws Exception {
        try (KVCacheManager.KVCacheSession session = new KVCacheManager.KVCacheSession(16, new BlockManager())) {
            session.allocate(config(), GenerationConfig.defaults());

            FlashAttentionKvScratch.KvPools pools =
                    FlashAttentionKvScratch.projectionScratchPools(session, 8, "test KV scratch");

            assertEquals(session.getWorkspace().getGateSeg().address(), pools.key().address());
            assertEquals(session.getWorkspace().getUpSeg().address(), pools.value().address());
            assertEquals(8L * Float.BYTES, pools.key().byteSize());
            assertEquals(8L * Float.BYTES, pools.value().byteSize());
        }
    }

    @Test
    void rejectsUnallocatedSession() {
        try (KVCacheManager.KVCacheSession session = new KVCacheManager.KVCacheSession(16, new BlockManager())) {
            assertThrows(IllegalArgumentException.class,
                    () -> FlashAttentionKvScratch.projectionScratchPools(session, 8, "test KV scratch"));
        }
    }

    @Test
    void rejectsEmptyPools() throws Exception {
        try (KVCacheManager.KVCacheSession session = new KVCacheManager.KVCacheSession(16, new BlockManager())) {
            session.allocate(config(), GenerationConfig.defaults());

            assertThrows(IllegalArgumentException.class,
                    () -> FlashAttentionKvScratch.projectionScratchPools(session, 0, "test KV scratch"));
        }
    }

    private static ModelConfig config() throws Exception {
        return OBJECT_MAPPER.readValue("""
                {
                  "model_type": "gemma4_text",
                  "hidden_size": 8,
                  "intermediate_size": 16,
                  "num_hidden_layers": 1,
                  "num_attention_heads": 2,
                  "num_key_value_heads": 1,
                  "head_dim": 4
                }
                """, ModelConfig.class);
    }
}
