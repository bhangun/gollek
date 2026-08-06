/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;
import tech.kayys.alkhawarizm.spi.model.ModelArchitecture;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;

import java.util.Map;

record DirectForwardPrefillRequest(
        AccelTensor embeddings,
        long[] inputIds,
        AccelTensor[] perLayerInputs,
        Map<String, AccelTensor> weights,
        ModelConfig config,
        ModelArchitecture arch,
        KVCacheManager.KVCacheSession kvCache,
        ResolvedModelWeights resolvedWeights,
        boolean embeddingsAlreadyInWorkspace) {

    static DirectForwardPrefillRequest tokenIds(
            long[] inputIds,
            Map<String, AccelTensor> weights,
            ModelConfig config,
            ModelArchitecture arch,
            KVCacheManager.KVCacheSession kvCache,
            ResolvedModelWeights resolvedWeights) {
        return new DirectForwardPrefillRequest(
                null, inputIds, null, weights, config, arch, kvCache, resolvedWeights, false);
    }

    static DirectForwardPrefillRequest embeddings(
            AccelTensor embeddings,
            long[] inputIds,
            AccelTensor[] perLayerInputs,
            Map<String, AccelTensor> weights,
            ModelConfig config,
            ModelArchitecture arch,
            KVCacheManager.KVCacheSession kvCache,
            ResolvedModelWeights resolvedWeights,
            boolean embeddingsAlreadyInWorkspace) {
        return new DirectForwardPrefillRequest(
                embeddings,
                inputIds,
                perLayerInputs,
                weights,
                config,
                arch,
                kvCache,
                resolvedWeights,
                embeddingsAlreadyInWorkspace);
    }

    DirectForwardPrefillRequest withPreparedTokenPrefill(
            DirectForwardInputPreparation.PreparedTokenPrefill prepared) {
        return embeddings(
                prepared.embeddings(),
                inputIds,
                prepared.perLayerInputs(),
                weights,
                config,
                arch,
                kvCache,
                resolvedWeights,
                true);
    }
}
