/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;

record DirectForwardMetalLinearWeightPlan(
        AccelTensor firstWeight,
        AccelTensor secondWeight,
        boolean nativeBf16Weights,
        boolean allowBf16ToF16Weights) {

    static DirectForwardMetalLinearWeightPlan single(
            ModelConfigTraits traits,
            String profileKey,
            boolean decodeLogitsPhase,
            long rows,
            AccelTensor weight) {
        boolean allowBf16ToF16Weight =
                allowBf16ToF16Weights(rows, traits, profileKey, decodeLogitsPhase);
        boolean nativeBf16Weight =
                nativeBf16Weights(traits, profileKey, allowBf16ToF16Weight, weight);
        return single(traits, weight, nativeBf16Weight, allowBf16ToF16Weight);
    }

    static DirectForwardMetalLinearWeightPlan single(
            ModelConfigTraits traits,
            AccelTensor weight,
            boolean nativeBf16Weight,
            boolean allowBf16ToF16Weight) {
        return new DirectForwardMetalLinearWeightPlan(
                toMetalHalfWeight(weight, traits, nativeBf16Weight, allowBf16ToF16Weight),
                null,
                nativeBf16Weight,
                allowBf16ToF16Weight);
    }

    static DirectForwardMetalLinearWeightPlan pair(
            ModelConfigTraits traits,
            String profileKey,
            boolean decodeLogitsPhase,
            long rows,
            AccelTensor firstWeight,
            AccelTensor secondWeight) {
        boolean allowBf16ToF16Weights =
                allowBf16ToF16Weights(rows, traits, profileKey, decodeLogitsPhase);
        boolean nativeBf16Weights =
                nativeBf16Weights(traits, profileKey, allowBf16ToF16Weights, firstWeight, secondWeight);
        return pair(traits, firstWeight, secondWeight, nativeBf16Weights, allowBf16ToF16Weights);
    }

    static DirectForwardMetalLinearWeightPlan pair(
            ModelConfigTraits traits,
            AccelTensor firstWeight,
            AccelTensor secondWeight,
            boolean nativeBf16Weights,
            boolean allowBf16ToF16Weights) {
        return new DirectForwardMetalLinearWeightPlan(
                toMetalHalfWeight(firstWeight, traits, nativeBf16Weights, allowBf16ToF16Weights),
                toMetalHalfWeight(secondWeight, traits, nativeBf16Weights, allowBf16ToF16Weights),
                nativeBf16Weights,
                allowBf16ToF16Weights);
    }

    static boolean allowBf16ToF16Weights(
            long rows,
            ModelConfigTraits traits,
            String profileKey,
            boolean decodeLogitsPhase) {
        return DirectForwardMetalLinearPolicy.allowNativeBf16ToF16ConversionForRows(
                rows,
                traits,
                profileKey,
                decodeLogitsPhase);
    }

    static boolean nativeBf16Weights(
            ModelConfigTraits traits,
            String profileKey,
            boolean allowBf16ToF16Weights,
            AccelTensor... weights) {
        if (weights == null || weights.length == 0) {
            return false;
        }
        for (AccelTensor weight : weights) {
            if (!DirectForwardMetalLinearPolicy.shouldUseNativeMetalBf16Linear(
                    weight, traits, profileKey, allowBf16ToF16Weights)) {
                return false;
            }
        }
        return true;
    }

    AccelTensor weight() {
        return firstWeight;
    }

    boolean completeSingle() {
        return firstWeight != null;
    }

    boolean completePair() {
        return firstWeight != null && secondWeight != null;
    }

    boolean hasUniformHalfOrBf16Weights() {
        if (!completeSingle()) {
            return false;
        }
        AccelTensor.QuantType weightType = firstWeight.quantType();
        if (secondWeight != null && secondWeight.quantType() != weightType) {
            return false;
        }
        return weightType == AccelTensor.QuantType.F16 || weightType == AccelTensor.QuantType.BF16;
    }

    boolean hasUniformPairHalfOrBf16Weights() {
        return completePair() && hasUniformHalfOrBf16Weights();
    }

    String conversionFailureReason() {
        return "weight_conversion_failed:native_bf16=" + nativeBf16Weights;
    }

    static AccelTensor toMetalHalfWeight(
            AccelTensor weight,
            ModelConfigTraits traits,
            boolean nativeBf16Weights,
            boolean allowBf16ToF16Weights) {
        return DirectForwardLinearCachePolicy.toMetalHalfWeight(
                weight,
                nativeBf16Weights,
                traits.nativeBf16Matvec(),
                allowBf16ToF16Weights,
                DirectForwardMetalLinearPolicy.allowMetalBf16Linear(traits));
    }
}
