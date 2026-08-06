/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;

final class FlashAttentionMetalLinearWeights {
    boolean canUseMixedHalfPairWeight(AccelTensor weight, FlashAttentionModelPolicy modelPolicy) {
        if (weight == null || !weight.isContiguous()) {
            return false;
        }
        AccelTensor.QuantType quantType = weight.quantType();
        if (quantType == AccelTensor.QuantType.BF16 && modelPolicy.disallowBf16ToF16LinearConversion()) {
            return shouldUseNativeMetalBf16Linear(modelPolicy, weight);
        }
        return quantType == AccelTensor.QuantType.F16
                || (quantType == AccelTensor.QuantType.BF16 && allowMetalBf16Linear(modelPolicy));
    }

    AccelTensor toMetalHalfWeight(AccelTensor weight, boolean nativeBf16,
            FlashAttentionModelPolicy modelPolicy) {
        if (weight == null) {
            return null;
        }
        if (weight.quantType() == AccelTensor.QuantType.F16) {
            return weight;
        }
        if (nativeBf16 && weight.quantType() == AccelTensor.QuantType.BF16) {
            return weight;
        }
        if (weight.quantType() == AccelTensor.QuantType.BF16
                && modelPolicy.disallowBf16ToF16LinearConversion()) {
            return null;
        }
        if (weight.quantType() == AccelTensor.QuantType.BF16 && allowMetalBf16Linear(modelPolicy)) {
            return weight.toF16CachedUpTo(FlashAttentionRuntimeOptions.metalF16WeightCacheMaxBytes());
        }
        return null;
    }

    boolean shouldUseNativeMetalBf16Linear(FlashAttentionModelPolicy modelPolicy, AccelTensor... weights) {
        if (!preferNativeMetalBf16Linear(modelPolicy) || weights == null || weights.length == 0) {
            return false;
        }
        for (AccelTensor weight : weights) {
            if (weight == null || weight.quantType() != AccelTensor.QuantType.BF16) {
                return false;
            }
        }
        return true;
    }

    private boolean allowMetalBf16Linear(FlashAttentionModelPolicy modelPolicy) {
        if (modelPolicy.preferNativeMetalBf16Linear()) {
            if (FlashAttentionRuntimeOptions.disableGemma4MetalBf16Linear()) {
                return false;
            }
        }
        return FlashAttentionRuntimeOptions.allowMetalBf16Linear();
    }

    private boolean preferNativeMetalBf16Linear(FlashAttentionModelPolicy modelPolicy) {
        if (modelPolicy.preferNativeMetalBf16Linear()) {
            String explicit = FlashAttentionRuntimeOptions.enableGemma4MetalBf16LinearValue();
            if (explicit != null && !explicit.isBlank()) {
                return Boolean.parseBoolean(explicit) && allowMetalBf16Linear(modelPolicy);
            }
            return allowMetalBf16Linear(modelPolicy);
        }
        return false;
    }
}
