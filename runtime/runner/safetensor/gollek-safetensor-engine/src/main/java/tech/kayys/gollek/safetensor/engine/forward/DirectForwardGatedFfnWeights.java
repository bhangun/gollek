/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;

record DirectForwardGatedFfnWeights(
        AccelTensor gateW,
        AccelTensor gateB,
        AccelTensor upW,
        AccelTensor upB,
        AccelTensor downW,
        AccelTensor downB) {

    static DirectForwardGatedFfnWeights fromLayer(ResolvedLayerWeights layerWeights) {
        return new DirectForwardGatedFfnWeights(
                layerWeights.ffnGateWeight(),
                layerWeights.ffnGateBias(),
                layerWeights.ffnUpWeight(),
                layerWeights.ffnUpBias(),
                layerWeights.ffnDownWeight(),
                layerWeights.ffnDownBias());
    }
}
