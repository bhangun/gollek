/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;
import tech.kayys.gollek.safetensor.generation.GenerationConfig;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;

import java.util.function.Supplier;

/**
 * Creates and allocates KV-cache sessions while keeping timing/profile updates together.
 */
final class DirectGenerationSessionAllocator {
    private final Supplier<KVCacheManager> kvCacheManager;

    DirectGenerationSessionAllocator(Supplier<KVCacheManager> kvCacheManager) {
        this.kvCacheManager = kvCacheManager;
    }

    KVCacheManager.KVCacheSession createAllocated(ModelConfig config, GenerationConfig cfg,
            DirectGenerationTimings timings, InferenceProfile profile) {
        KVCacheManager.KVCacheSession session = kvCacheManager.get().createSession(cfg.maxKvCacheTokens());
        boolean allocated = false;
        try {
            allocate(session, config, cfg, timings, profile);
            allocated = true;
            return session;
        } finally {
            if (!allocated) {
                closeQuietly(session);
            }
        }
    }

    void allocate(KVCacheManager.KVCacheSession session, ModelConfig config, GenerationConfig cfg,
            DirectGenerationTimings timings, InferenceProfile profile) {
        long tAlloc0 = System.nanoTime();
        session.allocate(config, cfg);
        if (timings != null) {
            timings.recordSessionAllocate(System.nanoTime() - tAlloc0, profile);
        }
    }

    private static void closeQuietly(AutoCloseable closeable) {
        if (closeable == null) {
            return;
        }
        try {
            closeable.close();
        } catch (Exception ignored) {
        }
    }
}
