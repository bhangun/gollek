/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;

record ResolvedLayerWeights(
        AccelTensor queryWeight,
        AccelTensor keyWeight,
        AccelTensor valueWeight,
        AccelTensor outputWeight,
        AccelTensor queryBias,
        AccelTensor keyBias,
        AccelTensor valueBias,
        AccelTensor outputBias,
        AccelTensor attentionNormWeight,
        AccelTensor ffnGateWeight,
        AccelTensor ffnGateBias,
        AccelTensor ffnUpWeight,
        AccelTensor ffnUpBias,
        AccelTensor ffnDownWeight,
        AccelTensor ffnDownBias,
        AccelTensor preFfnNormWeight,
        AccelTensor postFfnNormWeight,
        AccelTensor queryNormWeight,
        AccelTensor keyNormWeight,
        AccelTensor postAttnNormWeight,
        AccelTensor perLayerInputGateWeight,
        AccelTensor perLayerProjectionWeight,
        AccelTensor postPerLayerInputNormWeight,
        AccelTensor layerScalarWeight,
        float layerScalarValue) {

    boolean hasLayerScalar() {
        return layerScalarWeight != null;
    }
}
