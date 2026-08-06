/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import static tech.kayys.gollek.safetensor.engine.forward.DirectForwardRuntimeOptions.parseOptionalBoolean;

record DirectForwardFfnFastPathOptions(
        Boolean nativeBf16FusedHalfFfnExplicit,
        boolean disableNativeBf16FusedHalfFfn,
        boolean disableMetalFusedFfn,
        boolean enableSiluGatedFusedFfn,
        boolean enableMetalGegluFusedFfn,
        Boolean enableMetalFusedFfnPrefill,
        int nativeBf16FusedFfnPrefillMinRows,
        Boolean enableMetalMatvecFfnPrefillRows,
        Boolean preferMetalMatvecFfnPrefillRows,
        int metalMatvecFfnPrefillMaxRows,
        Boolean enableMetalGegluMatvecFfn,
        Boolean enableMetalSwigluMatvecFfn,
        Boolean enableMetalGateUpMatvecFfn,
        boolean disableMetalMatvecFfn,
        boolean validateMetalMatvecFfn) {

    static final String ALLOW_NATIVE_BF16_FUSED_HALF_FFN_PROPERTY =
            "gollek.safetensor.allow_native_bf16_fused_half_ffn";
    private static final String DISABLE_NATIVE_BF16_FUSED_HALF_FFN_PROPERTY =
            "gollek.safetensor.disable_native_bf16_fused_half_ffn";
    private static final String DISABLE_METAL_FUSED_FFN_PROPERTY =
            "gollek.safetensor.disable_metal_fused_ffn";
    private static final String ENABLE_SILU_GATED_FUSED_FFN_PROPERTY =
            "gollek.safetensor.enable_silu_gated_fused_ffn";
    private static final String ENABLE_METAL_GEGLU_FUSED_FFN_PROPERTY =
            "gollek.safetensor.enable_metal_geglu_fused_ffn";
    static final String ENABLE_METAL_FUSED_FFN_PREFILL_PROPERTY =
            "gollek.safetensor.enable_metal_fused_ffn_prefill";
    static final String NATIVE_BF16_FUSED_FFN_PREFILL_MIN_ROWS_PROPERTY =
            "gollek.safetensor.native_bf16_fused_ffn_prefill_min_rows";
    private static final int DEFAULT_NATIVE_BF16_FUSED_FFN_PREFILL_MIN_ROWS = 2;
    static final String ENABLE_METAL_MATVEC_FFN_PREFILL_ROWS_PROPERTY =
            "gollek.safetensor.enable_metal_matvec_ffn_prefill_rows";
    static final String PREFER_METAL_MATVEC_FFN_PREFILL_ROWS_PROPERTY =
            "gollek.safetensor.prefer_metal_matvec_ffn_prefill_rows";
    static final String METAL_MATVEC_FFN_PREFILL_MAX_ROWS_PROPERTY =
            "gollek.safetensor.metal_matvec_ffn_prefill_max_rows";
    private static final int DEFAULT_METAL_MATVEC_FFN_PREFILL_MAX_ROWS = 16;
    static final String ENABLE_METAL_GEGLU_MATVEC_FFN_PROPERTY =
            "gollek.safetensor.enable_metal_geglu_matvec_ffn";
    static final String ENABLE_METAL_SWIGLU_MATVEC_FFN_PROPERTY =
            "gollek.safetensor.enable_metal_swiglu_matvec_ffn";
    static final String ENABLE_METAL_GATE_UP_MATVEC_FFN_PROPERTY =
            "gollek.safetensor.enable_metal_gate_up_matvec_ffn";
    private static final String DISABLE_METAL_MATVEC_FFN_PROPERTY =
            "gollek.safetensor.disable_metal_matvec_ffn";
    private static final String VALIDATE_METAL_MATVEC_FFN_PROPERTY =
            "gollek.safetensor.validate_metal_matvec_ffn";

    static DirectForwardFfnFastPathOptions fromSystemProperties() {
        return new DirectForwardFfnFastPathOptions(
                parseOptionalBoolean(System.getProperty(ALLOW_NATIVE_BF16_FUSED_HALF_FFN_PROPERTY)),
                Boolean.getBoolean(DISABLE_NATIVE_BF16_FUSED_HALF_FFN_PROPERTY),
                Boolean.getBoolean(DISABLE_METAL_FUSED_FFN_PROPERTY),
                Boolean.TRUE.equals(parseOptionalBoolean(System.getProperty(ENABLE_SILU_GATED_FUSED_FFN_PROPERTY))),
                Boolean.getBoolean(ENABLE_METAL_GEGLU_FUSED_FFN_PROPERTY),
                parseOptionalBoolean(System.getProperty(ENABLE_METAL_FUSED_FFN_PREFILL_PROPERTY)),
                Integer.getInteger(NATIVE_BF16_FUSED_FFN_PREFILL_MIN_ROWS_PROPERTY,
                        DEFAULT_NATIVE_BF16_FUSED_FFN_PREFILL_MIN_ROWS),
                parseOptionalBoolean(System.getProperty(ENABLE_METAL_MATVEC_FFN_PREFILL_ROWS_PROPERTY)),
                parseOptionalBoolean(System.getProperty(PREFER_METAL_MATVEC_FFN_PREFILL_ROWS_PROPERTY)),
                Integer.getInteger(METAL_MATVEC_FFN_PREFILL_MAX_ROWS_PROPERTY,
                        DEFAULT_METAL_MATVEC_FFN_PREFILL_MAX_ROWS),
                parseOptionalBoolean(System.getProperty(ENABLE_METAL_GEGLU_MATVEC_FFN_PROPERTY)),
                parseOptionalBoolean(System.getProperty(ENABLE_METAL_SWIGLU_MATVEC_FFN_PROPERTY)),
                parseOptionalBoolean(System.getProperty(ENABLE_METAL_GATE_UP_MATVEC_FFN_PROPERTY)),
                Boolean.getBoolean(DISABLE_METAL_MATVEC_FFN_PROPERTY),
                Boolean.getBoolean(VALIDATE_METAL_MATVEC_FFN_PROPERTY));
    }

    static DirectForwardFfnFastPathOptions defaults() {
        return new DirectForwardFfnFastPathOptions(
                null, false, false, false, false, null,
                DEFAULT_NATIVE_BF16_FUSED_FFN_PREFILL_MIN_ROWS, null, null,
                DEFAULT_METAL_MATVEC_FFN_PREFILL_MAX_ROWS, null, null, null, false, false);
    }

    DirectForwardFfnFastPathOptions withNativeBf16FusedHalfFfn(Boolean allow, boolean disable) {
        return new DirectForwardFfnFastPathOptions(
                allow,
                disable,
                disableMetalFusedFfn,
                enableSiluGatedFusedFfn,
                enableMetalGegluFusedFfn,
                enableMetalFusedFfnPrefill,
                nativeBf16FusedFfnPrefillMinRows,
                enableMetalMatvecFfnPrefillRows,
                preferMetalMatvecFfnPrefillRows,
                metalMatvecFfnPrefillMaxRows,
                enableMetalGegluMatvecFfn,
                enableMetalSwigluMatvecFfn,
                enableMetalGateUpMatvecFfn,
                disableMetalMatvecFfn,
                validateMetalMatvecFfn);
    }

    DirectForwardFfnFastPathOptions withMetalFusedFfn(
            boolean disabled, boolean siluGatedEnabled, boolean gegluEnabled) {
        return new DirectForwardFfnFastPathOptions(
                nativeBf16FusedHalfFfnExplicit,
                disableNativeBf16FusedHalfFfn,
                disabled,
                siluGatedEnabled,
                gegluEnabled,
                enableMetalFusedFfnPrefill,
                nativeBf16FusedFfnPrefillMinRows,
                enableMetalMatvecFfnPrefillRows,
                preferMetalMatvecFfnPrefillRows,
                metalMatvecFfnPrefillMaxRows,
                enableMetalGegluMatvecFfn,
                enableMetalSwigluMatvecFfn,
                enableMetalGateUpMatvecFfn,
                disableMetalMatvecFfn,
                validateMetalMatvecFfn);
    }

    DirectForwardFfnFastPathOptions withMetalFusedFfnPrefill(Boolean enabled, int minRows) {
        return new DirectForwardFfnFastPathOptions(
                nativeBf16FusedHalfFfnExplicit,
                disableNativeBf16FusedHalfFfn,
                disableMetalFusedFfn,
                enableSiluGatedFusedFfn,
                enableMetalGegluFusedFfn,
                enabled,
                minRows,
                enableMetalMatvecFfnPrefillRows,
                preferMetalMatvecFfnPrefillRows,
                metalMatvecFfnPrefillMaxRows,
                enableMetalGegluMatvecFfn,
                enableMetalSwigluMatvecFfn,
                enableMetalGateUpMatvecFfn,
                disableMetalMatvecFfn,
                validateMetalMatvecFfn);
    }

    DirectForwardFfnFastPathOptions withMetalMatvecFfnPrefillRows(Boolean enabled, int maxRows) {
        return withMetalMatvecFfnPrefillRows(enabled, maxRows, preferMetalMatvecFfnPrefillRows);
    }

    DirectForwardFfnFastPathOptions withMetalMatvecFfnPrefillRows(
            Boolean enabled, int maxRows, Boolean preferRows) {
        return new DirectForwardFfnFastPathOptions(
                nativeBf16FusedHalfFfnExplicit,
                disableNativeBf16FusedHalfFfn,
                disableMetalFusedFfn,
                enableSiluGatedFusedFfn,
                enableMetalGegluFusedFfn,
                enableMetalFusedFfnPrefill,
                nativeBf16FusedFfnPrefillMinRows,
                enabled,
                preferRows,
                maxRows,
                enableMetalGegluMatvecFfn,
                enableMetalSwigluMatvecFfn,
                enableMetalGateUpMatvecFfn,
                disableMetalMatvecFfn,
                validateMetalMatvecFfn);
    }

    DirectForwardFfnFastPathOptions withMetalMatvecFfn(
            Boolean gegluEnabled, Boolean swigluEnabled, Boolean gateUpEnabled, boolean disabled) {
        return new DirectForwardFfnFastPathOptions(
                nativeBf16FusedHalfFfnExplicit,
                disableNativeBf16FusedHalfFfn,
                disableMetalFusedFfn,
                enableSiluGatedFusedFfn,
                enableMetalGegluFusedFfn,
                enableMetalFusedFfnPrefill,
                nativeBf16FusedFfnPrefillMinRows,
                enableMetalMatvecFfnPrefillRows,
                preferMetalMatvecFfnPrefillRows,
                metalMatvecFfnPrefillMaxRows,
                gegluEnabled,
                swigluEnabled,
                gateUpEnabled,
                disabled,
                validateMetalMatvecFfn);
    }

    DirectForwardFfnFastPathOptions withValidateMetalMatvecFfn(boolean validate) {
        return new DirectForwardFfnFastPathOptions(
                nativeBf16FusedHalfFfnExplicit,
                disableNativeBf16FusedHalfFfn,
                disableMetalFusedFfn,
                enableSiluGatedFusedFfn,
                enableMetalGegluFusedFfn,
                enableMetalFusedFfnPrefill,
                nativeBf16FusedFfnPrefillMinRows,
                enableMetalMatvecFfnPrefillRows,
                preferMetalMatvecFfnPrefillRows,
                metalMatvecFfnPrefillMaxRows,
                enableMetalGegluMatvecFfn,
                enableMetalSwigluMatvecFfn,
                enableMetalGateUpMatvecFfn,
                disableMetalMatvecFfn,
                validate);
    }
}
