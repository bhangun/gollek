/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import static tech.kayys.gollek.safetensor.engine.forward.DirectForwardRuntimeOptions.parseOptionalBoolean;

import java.util.Objects;

record DirectForwardMetalLinearOptions(
        boolean experimentalMetalLinearEnabled,
        boolean disableExperimentalMetalBf16Linear,
        Boolean experimentalMetalBf16Linear,
        boolean preferNativeMetalBf16Linear,
        DirectForwardNativeBf16LinearOptions nativeBf16Options) {

    DirectForwardMetalLinearOptions {
        nativeBf16Options = Objects.requireNonNull(nativeBf16Options, "nativeBf16Options");
    }

    private static final String EXPERIMENTAL_METAL_LINEAR_PROPERTY =
            "gollek.safetensor.experimental_metal_linear";
    private static final String DISABLE_EXPERIMENTAL_METAL_LINEAR_PROPERTY =
            "gollek.safetensor.disable_experimental_metal_linear";
    private static final String EXPERIMENTAL_METAL_BF16_LINEAR_PROPERTY =
            "gollek.safetensor.experimental_metal_bf16_linear";
    private static final String DISABLE_EXPERIMENTAL_METAL_BF16_LINEAR_PROPERTY =
            "gollek.safetensor.disable_experimental_metal_bf16_linear";
    private static final String PREFER_NATIVE_METAL_BF16_LINEAR_PROPERTY =
            "gollek.safetensor.prefer_native_metal_bf16_linear";

    static DirectForwardMetalLinearOptions fromSystemProperties() {
        return new DirectForwardMetalLinearOptions(
                resolveExperimentalMetalLinearEnabled(),
                Boolean.getBoolean(DISABLE_EXPERIMENTAL_METAL_BF16_LINEAR_PROPERTY),
                parseOptionalBoolean(System.getProperty(EXPERIMENTAL_METAL_BF16_LINEAR_PROPERTY)),
                Boolean.getBoolean(PREFER_NATIVE_METAL_BF16_LINEAR_PROPERTY),
                DirectForwardNativeBf16LinearOptions.fromSystemProperties());
    }

    static DirectForwardMetalLinearOptions defaults() {
        return new DirectForwardMetalLinearOptions(
                true, false, null, false, DirectForwardNativeBf16LinearOptions.defaults());
    }

    DirectForwardMetalLinearOptions withExperimentalMetalLinear(boolean enabled) {
        return new DirectForwardMetalLinearOptions(
                enabled,
                disableExperimentalMetalBf16Linear,
                experimentalMetalBf16Linear,
                preferNativeMetalBf16Linear,
                nativeBf16Options);
    }

    DirectForwardMetalLinearOptions withGenericBf16(Boolean enable, boolean disable, boolean preferNative) {
        return new DirectForwardMetalLinearOptions(
                experimentalMetalLinearEnabled,
                disable,
                enable,
                preferNative,
                nativeBf16Options);
    }

    DirectForwardMetalLinearOptions withNativeBf16(Boolean enable, boolean disable) {
        return new DirectForwardMetalLinearOptions(
                experimentalMetalLinearEnabled,
                disableExperimentalMetalBf16Linear,
                experimentalMetalBf16Linear,
                preferNativeMetalBf16Linear,
                nativeBf16Options.withMetalBf16(enable, disable));
    }

    DirectForwardMetalLinearOptions withNativeBf16ToF16(Boolean enable, boolean disable) {
        return new DirectForwardMetalLinearOptions(
                experimentalMetalLinearEnabled,
                disableExperimentalMetalBf16Linear,
                experimentalMetalBf16Linear,
                preferNativeMetalBf16Linear,
                nativeBf16Options.withBf16ToF16(enable, disable));
    }

    DirectForwardMetalLinearOptions withNativeBf16ToF16Ffn(Boolean enable, boolean disable) {
        return new DirectForwardMetalLinearOptions(
                experimentalMetalLinearEnabled,
                disableExperimentalMetalBf16Linear,
                experimentalMetalBf16Linear,
                preferNativeMetalBf16Linear,
                nativeBf16Options.withBf16ToF16Ffn(enable, disable));
    }

    DirectForwardMetalLinearOptions withNativeBf16ToF16Logits(Boolean enable, boolean disable) {
        return new DirectForwardMetalLinearOptions(
                experimentalMetalLinearEnabled,
                disableExperimentalMetalBf16Linear,
                experimentalMetalBf16Linear,
                preferNativeMetalBf16Linear,
                nativeBf16Options.withBf16ToF16Logits(enable, disable));
    }

    private static boolean resolveExperimentalMetalLinearEnabled() {
        if (Boolean.getBoolean(DISABLE_EXPERIMENTAL_METAL_LINEAR_PROPERTY)) {
            return false;
        }
        String explicit = System.getProperty(EXPERIMENTAL_METAL_LINEAR_PROPERTY);
        if (explicit != null && !explicit.isBlank()) {
            return Boolean.parseBoolean(explicit);
        }
        // Default ON for Apple Silicon safetensor text path to avoid accidental CPU-only runs.
        return true;
    }
}
