/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.weights.WeightTensorResolver;
import tech.kayys.alkhawarizm.spi.model.ModelArchitecture;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;

import java.lang.foreign.ValueLayout;
import java.util.Map;
import java.util.Objects;

record ResolvedModelWeights(
        String archId,
        String modelType,
        int numLayers,
        boolean addOneRmsNorm,
        AccelTensor embedTokens,
        AccelTensor packedPleEmbeddings,
        AccelTensor pleProjection,
        AccelTensor pleProjectionNorm,
        AccelTensor finalNorm,
        AccelTensor lmHead,
        ResolvedLayerWeights[] layers) {

    static ResolvedModelWeights create(Map<String, AccelTensor> weights, ModelConfig config,
            ModelArchitecture arch, boolean addOneRmsNorm) {
        int numLayers = Math.max(0, config.getNumHiddenLayers());
        AccelTensor embedTokens = WeightTensorResolver.first(weights, arch.embedTokensWeightCandidates());
        AccelTensor lmHead = WeightTensorResolver.first(weights, arch.lmHeadWeightCandidates());
        if (lmHead == null && config.isTieWordEmbeddings()) {
            lmHead = embedTokens;
        }
        ResolvedLayerWeights[] layers = new ResolvedLayerWeights[numLayers];
        for (int i = 0; i < numLayers; i++) {
            AccelTensor preFfnNorm = WeightTensorResolver.first(weights, arch.layerPreFfnNormWeightCandidates(i));
            if (preFfnNorm == null) {
                preFfnNorm = WeightTensorResolver.first(weights, arch.layerFfnNormWeightCandidates(i));
            }
            AccelTensor layerScalar = WeightTensorResolver.first(weights, arch.layerScalarWeightCandidates(i));
            layers[i] = new ResolvedLayerWeights(
                    WeightTensorResolver.first(weights, arch.layerQueryWeightCandidates(i)),
                    WeightTensorResolver.first(weights, arch.layerKeyWeightCandidates(i)),
                    WeightTensorResolver.first(weights, arch.layerValueWeightCandidates(i)),
                    WeightTensorResolver.first(weights, arch.layerOutputWeightCandidates(i)),
                    WeightTensorResolver.first(weights, arch.layerQueryBiasCandidates(i)),
                    WeightTensorResolver.first(weights, arch.layerKeyBiasCandidates(i)),
                    WeightTensorResolver.first(weights, arch.layerValueBiasCandidates(i)),
                    WeightTensorResolver.first(weights, arch.layerOutputBiasCandidates(i)),
                    WeightTensorResolver.first(weights, arch.layerAttentionNormWeightCandidates(i)),
                    WeightTensorResolver.first(weights, arch.layerFfnGateWeightCandidates(i)),
                    WeightTensorResolver.first(weights, arch.layerFfnGateBiasCandidates(i)),
                    WeightTensorResolver.first(weights, arch.layerFfnUpWeightCandidates(i)),
                    WeightTensorResolver.first(weights, arch.layerFfnUpBiasCandidates(i)),
                    WeightTensorResolver.first(weights, arch.layerFfnDownWeightCandidates(i)),
                    WeightTensorResolver.first(weights, arch.layerFfnDownBiasCandidates(i)),
                    preFfnNorm,
                    WeightTensorResolver.first(weights, arch.layerPostFfnNormWeightCandidates(i)),
                    WeightTensorResolver.first(weights, arch.layerQueryNormWeightCandidates(i)),
                    WeightTensorResolver.first(weights, arch.layerKeyNormWeightCandidates(i)),
                    WeightTensorResolver.first(weights, arch.layerPostAttnNormWeightCandidates(i)),
                    WeightTensorResolver.first(weights, arch.layerPerLayerInputGateWeightCandidates(i)),
                    WeightTensorResolver.first(weights, arch.layerPerLayerProjectionWeightCandidates(i)),
                    WeightTensorResolver.first(weights, arch.layerPostPerLayerInputNormWeightCandidates(i)),
                    layerScalar,
                    cachedScalarValue(layerScalar));
        }
        return new ResolvedModelWeights(
                arch.id(),
                config.getModelType(),
                numLayers,
                addOneRmsNorm,
                embedTokens,
                WeightTensorResolver.first(weights, arch.embedTokensPerLayerWeightCandidates()),
                WeightTensorResolver.first(weights, arch.perLayerModelProjectionWeightCandidates()),
                WeightTensorResolver.first(weights, arch.perLayerProjectionNormWeightCandidates()),
                WeightTensorResolver.first(weights, arch.finalNormWeightCandidates()),
                lmHead,
                layers);
    }

    boolean matches(ModelConfig config, ModelArchitecture arch, boolean addOneRmsNorm) {
        return numLayers == Math.max(0, config.getNumHiddenLayers())
                && this.addOneRmsNorm == addOneRmsNorm
                && Objects.equals(archId, arch.id())
                && Objects.equals(modelType, config.getModelType());
    }

    ResolvedLayerWeights layer(int index) {
        return layers[index];
    }

    private static float cachedScalarValue(AccelTensor tensor) {
        if (tensor == null) {
            return 1.0f;
        }
        if (tensor.quantType() == AccelTensor.QuantType.F32) {
            return tensor.dataPtr().get(ValueLayout.JAVA_FLOAT, 0);
        }
        AccelTensor dequantized = tensor.dequantize();
        try {
            return dequantized.dataPtr().get(ValueLayout.JAVA_FLOAT, 0);
        } finally {
            if (dequantized != tensor && !dequantized.isClosed()) {
                dequantized.close();
            }
        }
    }
}
