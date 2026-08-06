/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import tech.kayys.alkhawarizm.spi.model.FFNActivationType;

record DirectForwardMetalFfnActivationPlan(FFNActivationType activationType) {

    static DirectForwardMetalFfnActivationPlan from(FFNActivationType activationType) {
        return new DirectForwardMetalFfnActivationPlan(activationType);
    }

    boolean silu() {
        return activationType == FFNActivationType.SILU;
    }

    boolean gelu() {
        return activationType == FFNActivationType.GELU;
    }

    boolean supported() {
        return silu() || gelu();
    }

    String unsupportedReason() {
        return "unsupported_activation:" + activationType;
    }

    String metalName() {
        if (silu()) {
            return "swiglu";
        }
        if (gelu()) {
            return "geglu";
        }
        throw new IllegalStateException("Unsupported Metal FFN activation: " + activationType);
    }

    String acceptDecision(boolean nativeBf16Weights) {
        return "accept:" + metalName() + ":native_bf16=" + nativeBf16Weights;
    }

    String fusedPathSuffix(boolean nativeBf16Weights) {
        return metalName() + "_fused" + precisionSuffix(nativeBf16Weights);
    }

    String matvecPathSuffix(boolean nativeBf16Weights) {
        return metalName() + "_matvec" + precisionSuffix(nativeBf16Weights);
    }

    String gateUpPathSuffix(boolean nativeBf16Weights) {
        return metalName() + "_gate_up_matvec" + precisionSuffix(nativeBf16Weights);
    }

    String fusedProfilerKey() {
        return silu() ? "ffn_fused_metal" : "ffn_geglu_fused_metal";
    }

    String matvecProfilerKey(boolean nativeBf16Weights) {
        if (nativeBf16Weights) {
            return silu() ? "ffn_swiglu_matvec_bf16" : "ffn_geglu_matvec_bf16";
        }
        return silu() ? "ffn_swiglu_matvec_metal" : "ffn_geglu_matvec_metal";
    }

    String fusedHalfCapabilityRejection(DirectForwardMetalCapabilities capabilities) {
        if (silu() && !capabilities.supportsSwigluFfnHalf()) {
            return "swiglu_symbol_unavailable";
        }
        if (gelu() && !capabilities.supportsGegluFfnHalf()) {
            return "geglu_symbol_unavailable";
        }
        return null;
    }

    String matvecCapabilityRejection(DirectForwardMetalCapabilities capabilities, boolean nativeBf16Weights) {
        if (nativeBf16Weights && silu() && !capabilities.supportsSwigluFfnMatvecBf16()) {
            return "swiglu_bf16_matvec_symbol_unavailable";
        }
        if (nativeBf16Weights && gelu() && !capabilities.supportsGegluFfnMatvecBf16()) {
            return "geglu_bf16_matvec_symbol_unavailable";
        }
        if (!nativeBf16Weights && silu() && !capabilities.supportsSwigluFfnMatvecHalf()) {
            return "swiglu_matvec_symbol_unavailable";
        }
        if (!nativeBf16Weights && gelu() && !capabilities.supportsGegluFfnMatvecHalf()) {
            return "geglu_matvec_symbol_unavailable";
        }
        return null;
    }

    String gateUpCapabilityRejection(DirectForwardMetalCapabilities capabilities, boolean nativeBf16Weights) {
        if (nativeBf16Weights && silu() && !capabilities.supportsSwigluGateUpMatvecBf16()) {
            return "swiglu_bf16_symbol_unavailable";
        }
        if (nativeBf16Weights && gelu() && !capabilities.supportsGegluGateUpMatvecBf16()) {
            return "geglu_bf16_symbol_unavailable";
        }
        if (!nativeBf16Weights && silu() && !capabilities.supportsSwigluGateUpMatvecHalf()) {
            return "swiglu_symbol_unavailable";
        }
        if (!nativeBf16Weights && gelu() && !capabilities.supportsGegluGateUpMatvecHalf()) {
            return "geglu_symbol_unavailable";
        }
        return null;
    }

    private static String precisionSuffix(boolean nativeBf16Weights) {
        return nativeBf16Weights ? "_bf16" : "_f16";
    }
}
