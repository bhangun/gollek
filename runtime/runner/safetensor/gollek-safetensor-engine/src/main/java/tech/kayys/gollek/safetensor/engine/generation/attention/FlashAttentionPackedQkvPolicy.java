/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;

final class FlashAttentionPackedQkvPolicy {
    private FlashAttentionPackedQkvPolicy() {
    }

    static boolean shouldUse(AttentionInput in, ModelConfig config, int numQueryHeads, int numKeyValueHeads,
            FlashAttentionModelPolicy modelPolicy) {
        if (in == null) {
            return false;
        }
        if (modelPolicy != null && modelPolicy.packedQkvProjection()) {
            return true;
        }
        return looksLikeSharedPackedQkv(in, config, numQueryHeads, numKeyValueHeads);
    }

    static int resolveHeadDim(AccelTensor packedWeight, ModelConfig config, int numQueryHeads, int numKeyValueHeads) {
        int configured = config.resolvedHeadDim();
        int denominator = projectionDenominator(numQueryHeads, numKeyValueHeads);
        long expectedRows = (long) denominator * configured;
        if (configured > 0 && packedWeight != null && packedWeight.size(0) == expectedRows) {
            return configured;
        }

        if (packedWeight != null && denominator > 0 && packedWeight.size(0) % denominator == 0) {
            return Math.toIntExact(packedWeight.size(0) / denominator);
        }
        return configured;
    }

    private static boolean looksLikeSharedPackedQkv(AttentionInput in, ModelConfig config, int numQueryHeads,
            int numKeyValueHeads) {
        if (!sharesProjectionWeight(in)) {
            return false;
        }
        int denominator = projectionDenominator(numQueryHeads, numKeyValueHeads);
        if (denominator <= 0) {
            return false;
        }
        int configured = config.resolvedHeadDim();
        long rows = in.qW.size(0);
        return (configured > 0 && rows == (long) denominator * configured)
                || rows % denominator == 0;
    }

    private static int projectionDenominator(int numQueryHeads, int numKeyValueHeads) {
        return numQueryHeads + 2 * numKeyValueHeads;
    }

    private static boolean sharesProjectionWeight(AttentionInput in) {
        return in.qW != null && in.kW != null && in.vW != null && in.qW == in.kW && in.qW == in.vW;
    }
}
