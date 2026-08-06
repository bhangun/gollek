/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.safetensor.engine.generation.kv.BlockManager;
import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;
import tech.kayys.gollek.safetensor.generation.GenerationConfig;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DirectGenerationSessionAllocatorTest {

    @Test
    void createsAllocatesAndRecordsTiming() {
        TrackingSession session = new TrackingSession(false);
        TrackingManager manager = new TrackingManager(session);
        DirectGenerationSessionAllocator allocator = new DirectGenerationSessionAllocator(() -> manager);
        ModelConfig config = new ModelConfig();
        GenerationConfig cfg = GenerationConfig.builder().maxKvCacheTokens(123).build();
        DirectGenerationTimings timings = new DirectGenerationTimings();
        InferenceProfile profile = new InferenceProfile("test", false);

        KVCacheManager.KVCacheSession allocated =
                allocator.createAllocated(config, cfg, timings, profile);

        assertSame(session, allocated);
        assertEquals(123, manager.requestedMaxSeqLen);
        assertSame(config, session.config);
        assertSame(cfg, session.cfg);
        assertTrue(session.allocated);
        assertTrue(timings.benchTimings().sessionAllocateNanos() >= 0L);
        assertTrue(profile.sessionAllocateNanos >= 0L);
    }

    @Test
    void closesSessionWhenAllocationFails() {
        TrackingSession session = new TrackingSession(true);
        DirectGenerationSessionAllocator allocator =
                new DirectGenerationSessionAllocator(() -> new TrackingManager(session));

        assertThrows(IllegalStateException.class,
                () -> allocator.createAllocated(new ModelConfig(), GenerationConfig.defaults(),
                        new DirectGenerationTimings(), null));

        assertTrue(session.closed);
    }

    private static final class TrackingManager extends KVCacheManager {
        private final TrackingSession session;
        private int requestedMaxSeqLen;

        private TrackingManager(TrackingSession session) {
            this.session = session;
        }

        @Override
        public KVCacheSession createSession(int maxSeqLen) {
            requestedMaxSeqLen = maxSeqLen;
            return session;
        }
    }

    private static final class TrackingSession extends KVCacheManager.KVCacheSession {
        private final boolean failAllocate;
        private boolean allocated;
        private boolean closed;
        private ModelConfig config;
        private GenerationConfig cfg;

        private TrackingSession(boolean failAllocate) {
            super(1, new BlockManager());
            this.failAllocate = failAllocate;
        }

        @Override
        public void allocate(ModelConfig config, GenerationConfig genConfig) {
            this.config = config;
            this.cfg = genConfig;
            this.allocated = true;
            if (failAllocate) {
                throw new IllegalStateException("allocate failed");
            }
        }

        @Override
        public void close() {
            closed = true;
        }
    }
}
