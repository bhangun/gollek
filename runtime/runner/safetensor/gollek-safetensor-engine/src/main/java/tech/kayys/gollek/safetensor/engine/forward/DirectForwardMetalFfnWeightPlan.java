/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;

record DirectForwardMetalFfnWeightPlan(
        AccelTensor gateW,
        AccelTensor upW,
        AccelTensor downW,
        boolean nativeBf16Weights,
        boolean allowBf16ToF16Weights) {

    static DirectForwardMetalFfnWeightPlan gated(
            ModelConfigTraits traits,
            String profileKey,
            boolean decodeLogitsPhase,
            long rows,
            AccelTensor gateW,
            AccelTensor upW,
            AccelTensor downW) {
        boolean allowBf16ToF16Weights =
                allowBf16ToF16Weights(rows, traits, profileKey, decodeLogitsPhase);
        boolean nativeBf16Weights =
                nativeBf16Weights(traits, profileKey, allowBf16ToF16Weights, gateW, upW, downW);
        return gated(traits, gateW, upW, downW, nativeBf16Weights, allowBf16ToF16Weights);
    }

    static DirectForwardMetalFfnWeightPlan gated(
            ModelConfigTraits traits,
            AccelTensor gateW,
            AccelTensor upW,
            AccelTensor downW,
            boolean nativeBf16Weights,
            boolean allowBf16ToF16Weights) {
        return new DirectForwardMetalFfnWeightPlan(
                toMetalHalfWeight(gateW, traits, nativeBf16Weights, allowBf16ToF16Weights),
                toMetalHalfWeight(upW, traits, nativeBf16Weights, allowBf16ToF16Weights),
                toMetalHalfWeight(downW, traits, nativeBf16Weights, allowBf16ToF16Weights),
                nativeBf16Weights,
                allowBf16ToF16Weights);
    }

    static DirectForwardMetalFfnWeightPlan gateUp(
            ModelConfigTraits traits,
            AccelTensor gateW,
            AccelTensor upW,
            boolean nativeBf16Weights,
            boolean allowBf16ToF16Weights) {
        return new DirectForwardMetalFfnWeightPlan(
                toMetalHalfWeight(gateW, traits, nativeBf16Weights, allowBf16ToF16Weights),
                toMetalHalfWeight(upW, traits, nativeBf16Weights, allowBf16ToF16Weights),
                null,
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

    boolean completeGated() {
        return gateW != null && upW != null && downW != null;
    }

    boolean completeGateUp() {
        return gateW != null && upW != null;
    }

    boolean hasUniformHalfOrBf16Weights() {
        if (!completeGateUp()) {
            return false;
        }
        AccelTensor.QuantType weightType = gateW.quantType();
        if (upW.quantType() != weightType || (downW != null && downW.quantType() != weightType)) {
            return false;
        }
        return weightType == AccelTensor.QuantType.F16 || weightType == AccelTensor.QuantType.BF16;
    }

    String gatedConversionFailureReason() {
        return completeGated() ? null : conversionFailureReason();
    }

    String gatedUniformTypeFailureReason() {
        return hasUniformHalfOrBf16Weights() ? null : weightTypeMismatchReason();
    }

    String gateUpConversionFailureReason() {
        return completeGateUp() && hasUniformHalfOrBf16Weights() ? null : conversionFailureReason();
    }

    private String conversionFailureReason() {
        return "weight_conversion_failed:native_bf16=" + nativeBf16Weights;
    }

    private String weightTypeMismatchReason() {
        return "weight_type_mismatch:native_bf16=" + nativeBf16Weights;
    }

    private static AccelTensor toMetalHalfWeight(
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
