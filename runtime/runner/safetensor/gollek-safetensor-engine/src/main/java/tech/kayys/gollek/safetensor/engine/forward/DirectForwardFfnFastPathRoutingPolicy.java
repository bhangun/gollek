/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import static tech.kayys.gollek.safetensor.engine.forward.DirectForwardRuntimeOptions.runtimeOptionalBooleanProperty;

import java.util.Objects;

record DirectForwardFfnFastPathRoutingPolicy(DirectForwardFfnFastPathOptions options) {

    DirectForwardFfnFastPathRoutingPolicy {
        options = Objects.requireNonNull(options, "options");
    }

    static DirectForwardFfnFastPathRoutingPolicy from(DirectForwardFfnFastPathOptions options) {
        return new DirectForwardFfnFastPathRoutingPolicy(options);
    }

    boolean shouldUseMetalGegluFusedFfn(ModelConfigTraits traits) {
        return isNativeBf16FfnWithPerLayerInputTarget(traits) || options.enableMetalGegluFusedFfn();
    }

    boolean shouldTryLocalFusedHalfFfn(ModelConfigTraits traits) {
        return !isNativeBf16FfnWithPerLayerInputTarget(traits)
                || Boolean.TRUE.equals(allowNativeBf16FusedHalfFfnExplicit());
    }

    boolean isNativeBf16FusedHalfFfnAllowed() {
        if (options.disableNativeBf16FusedHalfFfn()) {
            return false;
        }
        Boolean explicit = allowNativeBf16FusedHalfFfnExplicit();
        if (explicit != null) {
            return explicit;
        }
        return false;
    }

    boolean allowNativeBf16FusedHalfFfn(long rows, ModelConfigTraits traits) {
        if (!isNativeBf16FfnWithPerLayerInputTarget(traits)) {
            return true;
        }
        if (options.disableNativeBf16FusedHalfFfn()) {
            return false;
        }
        Boolean explicit = allowNativeBf16FusedHalfFfnExplicit();
        if (explicit != null) {
            return explicit;
        }
        Boolean prefillExplicit = enableMetalFusedFfnPrefillExplicit();
        if (prefillExplicit != null) {
            return rows > 1L && prefillExplicit;
        }
        long effectiveMinRows = Math.max(2L, (long) nativeBf16FusedFfnPrefillMinRows());
        return rows >= effectiveMinRows && shouldUseMetalFusedFfnPrefill(traits);
    }

    boolean shouldUseMetalFusedFfnPrefill(ModelConfigTraits traits) {
        Boolean explicit = enableMetalFusedFfnPrefillExplicit();
        if (explicit != null) {
            return explicit;
        }
        // Prefer the fused GeGLU/SwiGLU prefill path for models with native BF16
        // matvec or per-layer embeddings: one dispatch replaces gate/up projection,
        // activation, and down projection orchestration across prompt rows.
        return isNativeBf16FfnWithPerLayerInputTarget(traits);
    }

    boolean shouldUseMetalGegluMatvecFfn(ModelConfigTraits traits) {
        if (options.disableMetalMatvecFfn()) {
            return false;
        }
        Boolean explicit = enableMetalGegluMatvecFfnExplicit();
        if (explicit != null) {
            return explicit;
        }
        // Decode is memory-bound: the native GeGLU matvec avoids three Java-orchestrated
        // launches for gate/up, activation, and down projection.
        return traits.geluGatedFfn() || isNativeBf16FfnWithPerLayerInputTarget(traits);
    }

    boolean shouldUseMetalSwigluMatvecFfn(ModelConfigTraits traits) {
        if (options.disableMetalMatvecFfn()) {
            return false;
        }
        Boolean explicit = enableMetalSwigluMatvecFfnExplicit();
        if (explicit != null) {
            return explicit;
        }
        // Any model declaring SILU gated activation (Qwen, IBM Granite, etc.) benefits
        // from the native single-token SwiGLU FFN path. The caller still rejects prefill rows.
        return traits.siluGated();
    }

    boolean shouldUseMetalGateUpMatvecFfn() {
        if (options.disableMetalMatvecFfn()) {
            return false;
        }
        return Boolean.TRUE.equals(enableMetalGateUpMatvecFfnExplicit());
    }

    boolean shouldUseMetalMatvecFfnPrefillRows(ModelConfigTraits traits, long rows) {
        if (options.disableMetalMatvecFfn() || !isNativeBf16FfnWithPerLayerInputTarget(traits)) {
            return false;
        }
        if (!Boolean.TRUE.equals(enableMetalMatvecFfnPrefillRowsExplicit())) {
            return false;
        }
        long maxRows = Math.max(0L, (long) metalMatvecFfnPrefillMaxRows());
        return rows > 1L && rows <= maxRows;
    }

    boolean shouldPreferMetalFusedFfnPrefillOverMatvecRows(ModelConfigTraits traits, long rows) {
        return rows > 1L
                && isNativeBf16FfnWithPerLayerInputTarget(traits)
                && !Boolean.TRUE.equals(preferMetalMatvecFfnPrefillRowsExplicit())
                && !options.disableMetalFusedFfn()
                && shouldUseMetalFusedFfnPrefill(traits)
                && allowNativeBf16FusedHalfFfn(rows, traits);
    }

    boolean shouldValidateMetalMatvecFfn(boolean traceFfnFastPath) {
        return options.validateMetalMatvecFfn() || traceFfnFastPath;
    }

    private Boolean allowNativeBf16FusedHalfFfnExplicit() {
        return runtimeOptionalBooleanProperty(
                DirectForwardFfnFastPathOptions.ALLOW_NATIVE_BF16_FUSED_HALF_FFN_PROPERTY,
                options.nativeBf16FusedHalfFfnExplicit());
    }

    private Boolean enableMetalFusedFfnPrefillExplicit() {
        return runtimeOptionalBooleanProperty(
                DirectForwardFfnFastPathOptions.ENABLE_METAL_FUSED_FFN_PREFILL_PROPERTY,
                options.enableMetalFusedFfnPrefill());
    }

    private Boolean enableMetalGegluMatvecFfnExplicit() {
        return runtimeOptionalBooleanProperty(
                DirectForwardFfnFastPathOptions.ENABLE_METAL_GEGLU_MATVEC_FFN_PROPERTY,
                options.enableMetalGegluMatvecFfn());
    }

    private Boolean enableMetalSwigluMatvecFfnExplicit() {
        return runtimeOptionalBooleanProperty(
                DirectForwardFfnFastPathOptions.ENABLE_METAL_SWIGLU_MATVEC_FFN_PROPERTY,
                options.enableMetalSwigluMatvecFfn());
    }

    private Boolean enableMetalGateUpMatvecFfnExplicit() {
        return runtimeOptionalBooleanProperty(
                DirectForwardFfnFastPathOptions.ENABLE_METAL_GATE_UP_MATVEC_FFN_PROPERTY,
                options.enableMetalGateUpMatvecFfn());
    }

    private Boolean enableMetalMatvecFfnPrefillRowsExplicit() {
        return runtimeOptionalBooleanProperty(
                DirectForwardFfnFastPathOptions.ENABLE_METAL_MATVEC_FFN_PREFILL_ROWS_PROPERTY,
                options.enableMetalMatvecFfnPrefillRows());
    }

    private Boolean preferMetalMatvecFfnPrefillRowsExplicit() {
        return runtimeOptionalBooleanProperty(
                DirectForwardFfnFastPathOptions.PREFER_METAL_MATVEC_FFN_PREFILL_ROWS_PROPERTY,
                options.preferMetalMatvecFfnPrefillRows());
    }

    private int nativeBf16FusedFfnPrefillMinRows() {
        String runtimeValue = System.getProperty(
                DirectForwardFfnFastPathOptions.NATIVE_BF16_FUSED_FFN_PREFILL_MIN_ROWS_PROPERTY);
        if (runtimeValue == null || runtimeValue.isBlank()) {
            return options.nativeBf16FusedFfnPrefillMinRows();
        }
        try {
            return Integer.parseInt(runtimeValue.trim());
        } catch (NumberFormatException ignored) {
            return options.nativeBf16FusedFfnPrefillMinRows();
        }
    }

    private int metalMatvecFfnPrefillMaxRows() {
        String runtimeValue = System.getProperty(
                DirectForwardFfnFastPathOptions.METAL_MATVEC_FFN_PREFILL_MAX_ROWS_PROPERTY);
        if (runtimeValue == null || runtimeValue.isBlank()) {
            return options.metalMatvecFfnPrefillMaxRows();
        }
        try {
            return Integer.parseInt(runtimeValue.trim());
        } catch (NumberFormatException ignored) {
            return options.metalMatvecFfnPrefillMaxRows();
        }
    }

    private static boolean isNativeBf16FfnWithPerLayerInputTarget(ModelConfigTraits traits) {
        return traits.nativeBf16Matvec() || traits.perLayerInputEmbedding();
    }
}
