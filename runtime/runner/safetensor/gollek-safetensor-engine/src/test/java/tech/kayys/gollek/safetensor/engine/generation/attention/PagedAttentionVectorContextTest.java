/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import org.junit.jupiter.api.Test;
import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.kv.BlockManager;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PagedAttentionVectorContextTest {

    @Test
    void derivesQueryOffsetsWindowAndGqaHeadMapping() {
        BlockManager blockManager = blockManager();
        try (AccelTensor query = AccelTensor.zeros(2, 3, 4, 5);
                AccelTensor output = AccelTensor.zeros(2, 3, 4, 5)) {
            PagedAttentionVectorContext context = PagedAttentionVectorContext.create(
                    query,
                    output,
                    List.of(7, 8),
                    blockManager,
                    PagedKvCacheLayout.packed(2, 5, 8),
                    BlockManager.KvStorageType.FP32,
                    8,
                    10,
                    2,
                    5,
                    0.25f,
                    true,
                    0.0f,
                    4,
                    true,
                    0);

            PagedAttentionVectorQuery firstHeadLastQuery = context.queryAt(0, 0, 2);
            PagedAttentionVectorQuery thirdHeadMiddleQuery = context.queryAt(1, 3, 1);

            assertEquals(0, firstHeadLastQuery.kvHeadIndex());
            assertEquals(9, firstHeadLastQuery.absolutePosition());
            assertEquals(6, firstHeadLastQuery.minPosition());
            assertEquals(2L * 4L * 5L * Float.BYTES, firstHeadLastQuery.queryByteOffset());
            assertEquals(2L * 4L * 5L, firstHeadLastQuery.outputElementIndex());
            assertTrue(firstHeadLastQuery.debugProbe());

            assertEquals(1, thirdHeadMiddleQuery.kvHeadIndex());
            assertEquals(8, thirdHeadMiddleQuery.absolutePosition());
            assertEquals(5, thirdHeadMiddleQuery.minPosition());
            assertFalse(thirdHeadMiddleQuery.debugProbe());
        } finally {
            blockManager.close();
        }
    }

    private static BlockManager blockManager() {
        BlockManager blockManager = new BlockManager();
        blockManager.initialize(8, 2, 5, 2);
        return blockManager;
    }
}
