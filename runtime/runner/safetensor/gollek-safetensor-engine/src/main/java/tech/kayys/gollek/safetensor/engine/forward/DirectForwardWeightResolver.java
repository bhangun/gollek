/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.alkhawarizm.spi.model.ModelArchitecture;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;

import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.Map;

final class DirectForwardWeightResolver {
    private final Map<Map<String, AccelTensor>, ResolvedModelWeights> resolvedWeightsCache =
            Collections.synchronizedMap(new IdentityHashMap<>());
    private volatile ResolvedWeightsCacheEntry lastResolvedWeights;

    void clear(Map<String, AccelTensor> weights) {
        if (weights == null) {
            return;
        }
        synchronized (resolvedWeightsCache) {
            resolvedWeightsCache.remove(weights);
        }
        ResolvedWeightsCacheEntry last = lastResolvedWeights;
        if (last != null && last.weights() == weights) {
            lastResolvedWeights = null;
        }
    }

    ResolvedModelWeights resolve(Map<String, AccelTensor> weights, ModelConfig config,
            ModelArchitecture arch, boolean addOneRmsNorm) {
        if (weights == null) {
            throw new IllegalArgumentException("weights must not be null");
        }
        ResolvedWeightsCacheEntry last = lastResolvedWeights;
        if (last != null && last.weights() == weights
                && last.config() == config
                && last.arch() == arch) {
            return last.resolved();
        }
        synchronized (resolvedWeightsCache) {
            ResolvedModelWeights cached = resolvedWeightsCache.get(weights);
            if (cached != null && cached.matches(config, arch, addOneRmsNorm)) {
                lastResolvedWeights = new ResolvedWeightsCacheEntry(weights, config, arch, cached);
                return cached;
            }
            ResolvedModelWeights resolved = ResolvedModelWeights.create(weights, config, arch, addOneRmsNorm);
            resolvedWeightsCache.put(weights, resolved);
            lastResolvedWeights = new ResolvedWeightsCacheEntry(weights, config, arch, resolved);
            return resolved;
        }
    }
}
