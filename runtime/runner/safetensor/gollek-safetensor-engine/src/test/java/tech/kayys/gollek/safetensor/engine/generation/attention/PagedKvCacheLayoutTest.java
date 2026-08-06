/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.safetensor.engine.generation.kv.BlockManager;

import static org.junit.jupiter.api.Assertions.assertEquals;

class PagedKvCacheLayoutTest {

    @Test
    void resolvesSourceCacheElementAndScaleOffsets() {
        BlockManager blockManager = new BlockManager();
        try {
            blockManager.initialize(16, 2, 4, 2);
            PagedKvCacheLayout layout = PagedKvCacheLayout.source(blockManager, 2, 4, 16);

            assertEquals(3, layout.tokenIndexInBlock(19));
            assertEquals(76, layout.sourceElement(1, 3));
            assertEquals(304, layout.sourceByteOffset(1, 3));
            assertEquals(19, layout.scaleIndex(1, 3));
        } finally {
            blockManager.close();
        }
    }

    @Test
    void resolvesTokenMajorAndBlockMajorDestinations() {
        PagedKvCacheLayout layout = PagedKvCacheLayout.packed(2, 4, 16);

        assertEquals(44, layout.tokenMajorElement(5, 1));
        assertEquals(332, layout.blockMajorElement(2, 1, 3));
        assertEquals(128, layout.blockElements());
    }
}
