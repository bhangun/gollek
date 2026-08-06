/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import org.junit.jupiter.api.Test;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.assertTrue;

class DirectLoadedModelReleaserTest {

    @Test
    void clearsWeightsAndClosesResources() {
        AtomicBoolean cacheCleared = new AtomicBoolean(false);
        CloseTracker weight = new CloseTracker(false);
        CloseTracker arena = new CloseTracker(false);
        Map<String, CloseTracker> weights = Map.of("w", weight);

        DirectLoadedModelReleaser.release(weights, arena, ignored -> cacheCleared.set(true));

        assertTrue(cacheCleared.get());
        assertTrue(weight.closed());
        assertTrue(arena.closed());
    }

    @Test
    void continuesReleaseWhenCleanupStepsThrow() {
        CloseTracker firstWeight = new CloseTracker(true);
        CloseTracker secondWeight = new CloseTracker(false);
        CloseTracker arena = new CloseTracker(true);
        Map<String, CloseTracker> weights = new LinkedHashMap<>();
        weights.put("first", firstWeight);
        weights.put("second", secondWeight);

        DirectLoadedModelReleaser.release(weights, arena, ignored -> {
            throw new IllegalStateException("cache clear failed");
        });

        assertTrue(firstWeight.closed());
        assertTrue(secondWeight.closed());
        assertTrue(arena.closed());
    }

    private static final class CloseTracker implements AutoCloseable {
        private final boolean fail;
        private boolean closed;

        private CloseTracker(boolean fail) {
            this.fail = fail;
        }

        @Override
        public void close() {
            closed = true;
            if (fail) {
                throw new IllegalStateException("close failed");
            }
        }

        private boolean closed() {
            return closed;
        }
    }
}
