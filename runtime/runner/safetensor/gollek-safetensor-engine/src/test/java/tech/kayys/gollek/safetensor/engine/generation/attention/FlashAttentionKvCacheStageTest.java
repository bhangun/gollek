/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import org.junit.jupiter.api.Test;
import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;

import static org.junit.jupiter.api.Assertions.assertTrue;

class FlashAttentionKvCacheStageTest {
    @Test
    void releasesSharedKvOwnedViewsWithoutClosingBackingState() {
        FlashAttentionKvCacheStage cacheStage = new FlashAttentionKvCacheStage();
        SharedKvState sharedState = new SharedKvState(
                AccelTensor.zeros(1, 1, 1, 2),
                AccelTensor.zeros(1, 1, 1, 2));
        sharedState.append(
                AccelTensor.zeros(1, 1, 1, 2),
                AccelTensor.zeros(1, 1, 1, 2));
        AccelTensor keyView = sharedState.key();
        AccelTensor valueView = sharedState.value();

        try {
            FlashAttentionKvCacheStage.CachedTensors cached =
                    new FlashAttentionKvCacheStage.CachedTensors(keyView, valueView, sharedState);

            cacheStage.releaseKeyValueViews(
                    new FlashAttentionKvCacheStage.State(false, null, false, true),
                    cached);

            assertTrue(keyView.isClosed());
            assertTrue(valueView.isClosed());
        } finally {
            sharedState.close();
        }
    }

    @Test
    void releasesPreparedSourceTensorsWhenSharedStateDidNotTakeOwnership() {
        FlashAttentionKvCacheStage cacheStage = new FlashAttentionKvCacheStage();
        AccelTensor key = AccelTensor.zeros(1, 1, 1, 2);
        AccelTensor value = AccelTensor.zeros(1, 1, 1, 2);

        cacheStage.releasePreparedKeyValueTensors(
                new FlashAttentionKvCacheStage.State(false, null, false, true),
                key,
                value);

        assertTrue(key.isClosed());
        assertTrue(value.isClosed());
    }
}
