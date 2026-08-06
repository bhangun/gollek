/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import tech.kayys.gollek.safetensor.engine.generation.kv.ForwardWorkspace;
import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;

import java.lang.foreign.MemorySegment;

/**
 * Provides reusable KV materialization scratch backed by the session forward
 * workspace so attention fallbacks avoid per-dispatch temporary arenas.
 */
final class FlashAttentionKvScratch {
    private FlashAttentionKvScratch() {
    }

    static KvPools projectionScratchPools(KVCacheManager.KVCacheSession kvSession, long poolElements,
            String purpose) {
        if (kvSession == null || kvSession.getWorkspace() == null) {
            throw new IllegalArgumentException(purpose + " requires an allocated KV session.");
        }
        if (poolElements <= 0L) {
            throw new IllegalArgumentException(purpose + " requires at least one element.");
        }
        long poolBytes = Math.multiplyExact(poolElements, (long) Float.BYTES);
        ForwardWorkspace workspace = kvSession.getWorkspace();
        workspace.ensureProjectionScratchCapacity(poolBytes);
        return new KvPools(
                workspace.getGateSeg().asSlice(0, poolBytes),
                workspace.getUpSeg().asSlice(0, poolBytes));
    }

    record KvPools(MemorySegment key, MemorySegment value) {
    }
}
