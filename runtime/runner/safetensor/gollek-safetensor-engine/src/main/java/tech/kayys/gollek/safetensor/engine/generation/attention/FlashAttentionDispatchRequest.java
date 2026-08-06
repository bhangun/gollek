/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;

import java.lang.foreign.MemorySegment;

record FlashAttentionDispatchRequest(AccelTensor query, AccelTensor key, AccelTensor value,
        KVCacheManager.KVCacheSession kvSession, SharedKvState sharedKvState, ModelConfig config,
        FlashAttentionModelPolicy modelPolicy, int layerIdx, int kvLayerIdx, int startPos, int seqLen,
        int numQueryHeads, int numKeyValueHeads, int headDim, float scale, boolean causal,
        float attentionSoftCap, MemorySegment attentionContextBuffer, boolean useDenseSharedKvState) {
}
