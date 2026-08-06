/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;

record DirectForwardAttentionWeights(
        AccelTensor queryWeight,
        AccelTensor keyWeight,
        AccelTensor valueWeight,
        AccelTensor outputWeight,
        AccelTensor queryBias,
        AccelTensor keyBias,
        AccelTensor valueBias,
        AccelTensor outputBias,
        AccelTensor attentionNormWeight,
        AccelTensor queryNormWeight,
        AccelTensor keyNormWeight,
        AccelTensor postAttnNormWeight) {

    static DirectForwardAttentionWeights fromLayer(ResolvedLayerWeights layerWeights) {
        return new DirectForwardAttentionWeights(
                layerWeights.queryWeight(),
                layerWeights.keyWeight(),
                layerWeights.valueWeight(),
                layerWeights.outputWeight(),
                layerWeights.queryBias(),
                layerWeights.keyBias(),
                layerWeights.valueBias(),
                layerWeights.outputBias(),
                layerWeights.attentionNormWeight(),
                layerWeights.queryNormWeight(),
                layerWeights.keyNormWeight(),
                layerWeights.postAttnNormWeight());
    }
}
