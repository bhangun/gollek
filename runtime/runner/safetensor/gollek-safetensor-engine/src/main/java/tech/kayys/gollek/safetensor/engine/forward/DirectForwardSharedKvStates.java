/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import tech.kayys.gollek.safetensor.engine.generation.attention.SharedKvState;
import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;

import java.util.Map;

final class DirectForwardSharedKvStates {
    private DirectForwardSharedKvStates() {
    }

    static Map<Integer, SharedKvState> forPrefill(
            ModelConfig config,
            KVCacheManager.KVCacheSession kvCache,
            int startPos) {
        if (!enabled(config)) {
            return null;
        }
        if (startPos == 0) {
            kvCache.clearSharedKvStates();
        }
        return kvCache.sharedKvStates();
    }

    static Map<Integer, SharedKvState> forDecode(
            ModelConfig config,
            KVCacheManager.KVCacheSession kvCache) {
        if (!enabled(config)) {
            return null;
        }
        return kvCache.sharedKvStates();
    }

    private static boolean enabled(ModelConfig config) {
        return config.getResolvedNumKvSharedLayers() > 0;
    }
}
