/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import java.util.Objects;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;

record DirectForwardMetalLinearRoutingPolicy(
        DirectForwardMetalLinearOptions options,
        DirectForwardNativeBf16LinearPolicy nativeBf16Policy) {

    DirectForwardMetalLinearRoutingPolicy {
        options = Objects.requireNonNull(options, "options");
        nativeBf16Policy = Objects.requireNonNull(nativeBf16Policy, "nativeBf16Policy");
    }

    static DirectForwardMetalLinearRoutingPolicy from(DirectForwardMetalLinearOptions options) {
        DirectForwardMetalLinearOptions resolved =
                options == null ? DirectForwardMetalLinearOptions.defaults() : options;
        return new DirectForwardMetalLinearRoutingPolicy(
                resolved,
                DirectForwardNativeBf16LinearPolicy.from(resolved.nativeBf16Options()));
    }

    boolean experimentalMetalLinearEnabled() {
        return options.experimentalMetalLinearEnabled();
    }

    boolean allowMetalBf16Linear(ModelConfigTraits traits) {
        if (traits.nativeBf16Matvec()) {
            return nativeBf16Policy.allowMetalBf16Linear(traits);
        }
        return allowGenericMetalBf16Linear();
    }

    boolean shouldUseNativeMetalBf16Linear(
            AccelTensor weight,
            ModelConfigTraits traits,
            String profileKey) {
        return shouldUseNativeMetalBf16Linear(
                weight,
                traits,
                profileKey,
                allowNativeBf16ToF16Conversion(traits, profileKey));
    }

    boolean shouldUseNativeMetalBf16Linear(
            AccelTensor weight,
            ModelConfigTraits traits,
            String profileKey,
            boolean allowBf16ToF16) {
        return preferNativeMetalBf16Linear(traits, profileKey, allowBf16ToF16)
                && allowMetalBf16Linear(traits)
                && weight != null
                && weight.quantType() == AccelTensor.QuantType.BF16;
    }

    boolean canUseMetalHalfLinearCandidate(
            boolean canUseExperimentalMetalLinear,
            AccelTensor input,
            AccelTensor weight,
            ModelConfigTraits traits,
            String profileKey) {
        if (!canUseExperimentalMetalLinear) {
            return false;
        }
        if (input == null || input.quantType() != AccelTensor.QuantType.F32) {
            return false;
        }
        if (weight == null) {
            return false;
        }
        AccelTensor.QuantType quantType = weight.quantType();
        if (quantType == AccelTensor.QuantType.BF16 && traits.nativeBf16Matvec()
                && !shouldUseNativeMetalBf16Linear(weight, traits, profileKey)
                && !allowNativeBf16ToF16Conversion(traits, profileKey)) {
            return false;
        }
        boolean isHalfCompatible = quantType == AccelTensor.QuantType.F16
                || (quantType == AccelTensor.QuantType.BF16 && allowMetalBf16Linear(traits))
                || quantType == AccelTensor.QuantType.NF4;
        boolean isQuantizedToF16 = quantType == AccelTensor.QuantType.INT4
                || quantType == AccelTensor.QuantType.INT8
                || quantType == AccelTensor.QuantType.FP8;
        if (!isHalfCompatible && !isQuantizedToF16) {
            return false;
        }
        if (!weight.isContiguous()) {
            return false;
        }
        int inputRank = input.rank();
        if (inputRank < 2 || weight.rank() != 2) {
            return false;
        }
        long rows = input.numel() / Math.max(1L, input.size(-1));
        if (rows <= 0L) {
            return false;
        }
        long batchProduct = 1L;
        for (int i = 0; i < inputRank - 2; i++) {
            batchProduct *= input.size(i);
        }
        return batchProduct == 1L;
    }

    boolean canUseMetalHalfWeight(AccelTensor weight, ModelConfigTraits traits, String profileKey) {
        if (weight == null || weight.rank() != 2 || !weight.isContiguous()) {
            return false;
        }
        AccelTensor.QuantType quantType = weight.quantType();
        if (quantType == AccelTensor.QuantType.BF16 && traits.nativeBf16Matvec()) {
            return shouldUseNativeMetalBf16Linear(weight, traits, profileKey)
                    || allowNativeBf16ToF16Conversion(traits, profileKey);
        }
        if (quantType == AccelTensor.QuantType.F16
                || (quantType == AccelTensor.QuantType.BF16 && allowMetalBf16Linear(traits))
                || quantType == AccelTensor.QuantType.NF4) {
            return true;
        }
        // Allow quantized weights: they will be converted to F16 on first use and cached.
        if (quantType == AccelTensor.QuantType.INT4
                || quantType == AccelTensor.QuantType.INT8
                || quantType == AccelTensor.QuantType.FP8) {
            return true;
        }
        return false;
    }


    boolean allowNativeBf16ToF16Conversion(ModelConfigTraits traits, String profileKey) {
        return nativeBf16Policy.allowBf16ToF16Linear(traits, profileKey);
    }

    boolean allowNativeBf16ToF16ConversionForRows(
            long rows,
            ModelConfigTraits traits,
            String profileKey,
            boolean decodeLogitsPhase) {
        return nativeBf16Policy.allowBf16ToF16LinearForRows(rows, traits, profileKey, decodeLogitsPhase);
    }

    private boolean allowGenericMetalBf16Linear() {
        if (options.disableExperimentalMetalBf16Linear()) {
            return false;
        }
        if (options.experimentalMetalBf16Linear() != null) {
            return options.experimentalMetalBf16Linear();
        }
        return false;
    }

    private boolean preferNativeMetalBf16Linear(
            ModelConfigTraits traits,
            String profileKey,
            boolean allowBf16ToF16) {
        if (traits.nativeBf16Matvec()) {
            if (allowBf16ToF16) {
                return false;
            }
            return nativeBf16Policy.preferNativeMetalBf16Linear(traits);
        }
        return options.preferNativeMetalBf16Linear();
    }
}
