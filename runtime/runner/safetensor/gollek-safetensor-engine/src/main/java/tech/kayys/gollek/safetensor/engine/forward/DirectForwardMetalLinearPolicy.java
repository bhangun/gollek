/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;

final class DirectForwardMetalLinearPolicy {
    private static final DirectForwardMetalLinearRoutingPolicy POLICY =
            DirectForwardMetalLinearRoutingPolicy.from(
                    DirectForwardMetalLinearOptions.fromSystemProperties());

    private DirectForwardMetalLinearPolicy() {
    }

    static boolean experimentalMetalLinearEnabled() {
        return POLICY.experimentalMetalLinearEnabled();
    }

    static boolean allowMetalBf16Linear(ModelConfigTraits traits) {
        return POLICY.allowMetalBf16Linear(traits);
    }

    static boolean shouldUseNativeMetalBf16Linear(
            AccelTensor weight,
            ModelConfigTraits traits,
            String profileKey) {
        return POLICY.shouldUseNativeMetalBf16Linear(weight, traits, profileKey);
    }

    static boolean shouldUseNativeMetalBf16Linear(
            AccelTensor weight,
            ModelConfigTraits traits,
            String profileKey,
            boolean allowBf16ToF16) {
        return POLICY.shouldUseNativeMetalBf16Linear(weight, traits, profileKey, allowBf16ToF16);
    }

    static boolean canUseMetalHalfLinearCandidate(
            boolean canUseExperimentalMetalLinear,
            AccelTensor input,
            AccelTensor weight,
            ModelConfigTraits traits,
            String profileKey) {
        return POLICY.canUseMetalHalfLinearCandidate(canUseExperimentalMetalLinear, input, weight, traits, profileKey);
    }

    static boolean canUseMetalHalfWeight(AccelTensor weight, ModelConfigTraits traits, String profileKey) {
        return POLICY.canUseMetalHalfWeight(weight, traits, profileKey);
    }

    static boolean allowNativeBf16ToF16Conversion(ModelConfigTraits traits, String profileKey) {
        return POLICY.allowNativeBf16ToF16Conversion(traits, profileKey);
    }

    static boolean allowNativeBf16ToF16ConversionForRows(
            long rows,
            ModelConfigTraits traits,
            String profileKey,
            boolean decodeLogitsPhase) {
        return POLICY.allowNativeBf16ToF16ConversionForRows(rows, traits, profileKey, decodeLogitsPhase);
    }
}
