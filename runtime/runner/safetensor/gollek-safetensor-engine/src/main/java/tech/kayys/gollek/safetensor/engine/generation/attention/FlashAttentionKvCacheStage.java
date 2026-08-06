/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;

final class FlashAttentionKvCacheStage {
    State resolveState(AttentionInput in, ModelConfig config, int layerIdx, int kvLayerIdx) {
        boolean sharedKv = config.usesSharedKvCache(layerIdx);
        SharedKvState sharedKvState = sharedKv && in.sharedKvStates != null
                ? in.sharedKvStates.get(kvLayerIdx)
                : null;
        boolean useDenseSharedKvState = sharedKvState != null;
        boolean storeSharedKvState = !sharedKv
                && in.sharedKvStates != null
                && config.isSharedKvSourceLayer(layerIdx);
        return new State(sharedKv, sharedKvState, useDenseSharedKvState, storeSharedKvState);
    }

    CachedTensors updateCache(AttentionInput in, State state, KVCacheManager.KVCacheSession kvSession,
            AccelTensor key, AccelTensor value, int seqLen, int numKeyValueHeads, int headDim) {
        if (!state.sharedKv()) {
            PagedKvCacheIO.updateCache(key, value, kvSession, in.layerIdx, in.startPos, seqLen, numKeyValueHeads,
                    headDim);
        }
        if (!state.storeSharedKvState()) {
            return new CachedTensors(key, value, null);
        }

        SharedKvState appended = appendSharedKvState(in.sharedKvStates.get(in.layerIdx), key, value);
        in.sharedKvStates.put(in.layerIdx, appended);
        return new CachedTensors(appended.key(), appended.value(), appended);
    }

    void releaseKeyValueViews(State state, CachedTensors cached) {
        if (cached == null) {
            return;
        }
        if (cached.viewOwner() != null) {
            cached.viewOwner().releaseView(cached.key());
            cached.viewOwner().releaseView(cached.value());
            return;
        }
        releasePreparedKeyValueTensors(state, cached.key(), cached.value());
    }

    void releasePreparedKeyValueTensors(State state, AccelTensor key, AccelTensor value) {
        if (state.useDenseSharedKvState() && state.sharedKvState() != null) {
            state.sharedKvState().releaseView(key);
            state.sharedKvState().releaseView(value);
            return;
        }
        closeIfOpen(key);
        closeIfOpen(value);
    }

    private void closeIfOpen(AccelTensor tensor) {
        if (tensor != null && !tensor.isClosed()) {
            tensor.close();
        }
    }

    private SharedKvState appendSharedKvState(SharedKvState existing, AccelTensor deltaKey, AccelTensor deltaValue) {
        if (existing == null) {
            return new SharedKvState(deltaKey, deltaValue);
        }
        existing.append(deltaKey, deltaValue);
        return existing;
    }

    record State(boolean sharedKv, SharedKvState sharedKvState, boolean useDenseSharedKvState,
            boolean storeSharedKvState) {
    }

    record CachedTensors(AccelTensor key, AccelTensor value, SharedKvState viewOwner) {
    }
}
