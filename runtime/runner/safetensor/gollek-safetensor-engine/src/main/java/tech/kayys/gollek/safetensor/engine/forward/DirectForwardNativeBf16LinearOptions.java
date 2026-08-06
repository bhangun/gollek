/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import static tech.kayys.gollek.safetensor.engine.forward.DirectForwardRuntimeOptions.parseOptionalBoolean;

/**
 * Configuration options for the native BF16 Metal linear projection path.
 *
 * <p>These options apply to any model that declares the {@code nativeBf16Matvec} capability
 * in its {@link tech.kayys.alkhawarizm.spi.model.ModelRuntimeTraits}. They are intentionally
 * model-family agnostic — policy classes must not check model identity.
 */
record DirectForwardNativeBf16LinearOptions(
        Boolean enableMetalBf16Linear,
        boolean disableMetalBf16Linear,
        Boolean enableBf16ToF16Linear,
        boolean disableBf16ToF16Linear,
        Boolean enableBf16ToF16FfnLinear,
        boolean disableBf16ToF16FfnLinear,
        Boolean enableBf16ToF16LogitsLinear,
        boolean disableBf16ToF16LogitsLinear) {

    private static final String ENABLE_METAL_BF16_LINEAR_PROPERTY =
            "gollek.safetensor.enable_metal_bf16_linear";
    private static final String DISABLE_METAL_BF16_LINEAR_PROPERTY =
            "gollek.safetensor.disable_metal_bf16_linear";
    private static final String ENABLE_BF16_TO_F16_LINEAR_PROPERTY =
            "gollek.safetensor.enable_bf16_to_f16_linear";
    private static final String DISABLE_BF16_TO_F16_LINEAR_PROPERTY =
            "gollek.safetensor.disable_bf16_to_f16_linear";
    private static final String ENABLE_BF16_TO_F16_FFN_LINEAR_PROPERTY =
            "gollek.safetensor.enable_bf16_to_f16_ffn_linear";
    private static final String DISABLE_BF16_TO_F16_FFN_LINEAR_PROPERTY =
            "gollek.safetensor.disable_bf16_to_f16_ffn_linear";
    private static final String ENABLE_BF16_TO_F16_LOGITS_LINEAR_PROPERTY =
            "gollek.safetensor.enable_bf16_to_f16_logits_linear";
    private static final String DISABLE_BF16_TO_F16_LOGITS_LINEAR_PROPERTY =
            "gollek.safetensor.disable_bf16_to_f16_logits_linear";

    static DirectForwardNativeBf16LinearOptions fromSystemProperties() {
        return new DirectForwardNativeBf16LinearOptions(
                parseOptionalBoolean(System.getProperty(ENABLE_METAL_BF16_LINEAR_PROPERTY)),
                Boolean.getBoolean(DISABLE_METAL_BF16_LINEAR_PROPERTY),
                parseOptionalBoolean(System.getProperty(ENABLE_BF16_TO_F16_LINEAR_PROPERTY)),
                Boolean.getBoolean(DISABLE_BF16_TO_F16_LINEAR_PROPERTY),
                parseOptionalBoolean(System.getProperty(ENABLE_BF16_TO_F16_FFN_LINEAR_PROPERTY)),
                Boolean.getBoolean(DISABLE_BF16_TO_F16_FFN_LINEAR_PROPERTY),
                parseOptionalBoolean(System.getProperty(ENABLE_BF16_TO_F16_LOGITS_LINEAR_PROPERTY)),
                Boolean.getBoolean(DISABLE_BF16_TO_F16_LOGITS_LINEAR_PROPERTY));
    }

    static DirectForwardNativeBf16LinearOptions defaults() {
        return new DirectForwardNativeBf16LinearOptions(null, false, null, false, null, false, null, false);
    }

    DirectForwardNativeBf16LinearOptions withMetalBf16(Boolean enable, boolean disable) {
        return new DirectForwardNativeBf16LinearOptions(
                enable,
                disable,
                enableBf16ToF16Linear,
                disableBf16ToF16Linear,
                enableBf16ToF16FfnLinear,
                disableBf16ToF16FfnLinear,
                enableBf16ToF16LogitsLinear,
                disableBf16ToF16LogitsLinear);
    }

    DirectForwardNativeBf16LinearOptions withBf16ToF16(Boolean enable, boolean disable) {
        return new DirectForwardNativeBf16LinearOptions(
                enableMetalBf16Linear,
                disableMetalBf16Linear,
                enable,
                disable,
                enableBf16ToF16FfnLinear,
                disableBf16ToF16FfnLinear,
                enableBf16ToF16LogitsLinear,
                disableBf16ToF16LogitsLinear);
    }

    DirectForwardNativeBf16LinearOptions withBf16ToF16Ffn(Boolean enable, boolean disable) {
        return new DirectForwardNativeBf16LinearOptions(
                enableMetalBf16Linear,
                disableMetalBf16Linear,
                enableBf16ToF16Linear,
                disableBf16ToF16Linear,
                enable,
                disable,
                enableBf16ToF16LogitsLinear,
                disableBf16ToF16LogitsLinear);
    }

    DirectForwardNativeBf16LinearOptions withBf16ToF16Logits(Boolean enable, boolean disable) {
        return new DirectForwardNativeBf16LinearOptions(
                enableMetalBf16Linear,
                disableMetalBf16Linear,
                enableBf16ToF16Linear,
                disableBf16ToF16Linear,
                enableBf16ToF16FfnLinear,
                disableBf16ToF16FfnLinear,
                enable,
                disable);
    }
}
