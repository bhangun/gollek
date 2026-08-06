/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.attention.AttentionInput;
import tech.kayys.gollek.safetensor.engine.generation.attention.SharedKvState;
import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;

import java.util.Map;

final class DirectForwardAttentionInputs {
    private DirectForwardAttentionInputs() {
    }

    static AttentionInput causalLayerInput(DirectForwardLayerStageContext ctx,
                                           DirectForwardAttentionWeights weights,
                                           AccelTensor normalizedInput,
                                           KVCacheManager.KVCacheSession kvCache,
                                           int startPos,
                                           Map<Integer, SharedKvState> sharedKvStates) {
        return new AttentionInput(
                normalizedInput,
                weights.queryWeight(),
                weights.keyWeight(),
                weights.valueWeight(),
                weights.outputWeight(),
                weights.queryBias(),
                weights.keyBias(),
                weights.valueBias(),
                weights.outputBias(),
                ctx.arch(),
                ctx.config(),
                kvCache,
                ctx.layerIdx(),
                startPos,
                true,
                weights.queryNormWeight(),
                weights.keyNormWeight(),
                weights.postAttnNormWeight(),
                sharedKvStates,
                ctx.hiddenOut(),
                ctx.workspace().getNormedFfnSeg());
    }
}
