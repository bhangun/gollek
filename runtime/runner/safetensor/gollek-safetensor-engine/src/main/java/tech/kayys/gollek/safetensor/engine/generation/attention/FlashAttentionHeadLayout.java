/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;

record FlashAttentionHeadLayout(int numQueryHeads, int numKeyValueHeads, int headDim,
        boolean packedQkvProjection) {

    static FlashAttentionHeadLayout resolve(AttentionInput in, ModelConfig config, int layerIdx) {
        FlashAttentionModelPolicy policy = FlashAttentionModelPolicy.resolve(in == null ? null : in.arch, config);
        return resolve(in, config, layerIdx, policy);
    }

    static FlashAttentionHeadLayout resolve(AttentionInput in, ModelConfig config, int layerIdx,
            FlashAttentionModelPolicy modelPolicy) {
        int numQueryHeads = config.getNumAttentionHeads();
        int numKeyValueHeads = config.getResolvedNumKvHeadsForLayer(layerIdx);
        boolean packedQkv = FlashAttentionPackedQkvPolicy.shouldUse(
                in, config, numQueryHeads, numKeyValueHeads, modelPolicy);
        AccelTensor queryWeight = in == null ? null : in.qW;
        int headDim = packedQkv
                ? FlashAttentionPackedQkvPolicy.resolveHeadDim(queryWeight, config, numQueryHeads, numKeyValueHeads)
                : resolveProjectedHeadDim(queryWeight, config, numQueryHeads, layerIdx);
        return new FlashAttentionHeadLayout(numQueryHeads, numKeyValueHeads, headDim, packedQkv);
    }

    int queryProjectionDim() {
        return numQueryHeads * headDim;
    }

    int keyValueProjectionDim() {
        return numKeyValueHeads * headDim;
    }

    int packedQkvProjectionDim() {
        return queryProjectionDim() + keyValueProjectionDim() + keyValueProjectionDim();
    }

    private static int resolveProjectedHeadDim(AccelTensor queryWeight, ModelConfig config,
            int numQueryHeads, int layerIdx) {
        FlashAttentionShapeAdmissionPlan admission =
                FlashAttentionShapeAdmissionPlan.separateQueryWeightLayout(
                        queryWeight, config, numQueryHeads, layerIdx);
        if (!admission.admitted()) {
            throw admission.asException();
        }
        return admission.resolvedHeadDim();
    }
}
