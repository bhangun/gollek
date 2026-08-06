/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import java.util.Objects;

import tech.kayys.alkhawarizm.metal.binding.MetalBinding;
import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;

/**
 * Describes the native BF16 row-batched Metal matvec FFN kernel used by the
 * short-prefill experiment.
 */
record DirectForwardMetalMatvecRowsGatedFfnKernelPlan(
        DirectForwardMetalFfnActivationPlan activation) {

    DirectForwardMetalMatvecRowsGatedFfnKernelPlan {
        Objects.requireNonNull(activation, "activation");
        if (!activation.supported()) {
            throw new IllegalArgumentException("Unsupported matvec rows gated FFN activation: "
                    + activation.activationType());
        }
    }

    static DirectForwardMetalMatvecRowsGatedFfnKernelPlan from(
            DirectForwardMetalFfnActivationPlan activation) {
        return new DirectForwardMetalMatvecRowsGatedFfnKernelPlan(activation);
    }

    String pathSuffix(int nativeVariant) {
        return activation.metalName() + "_matvec_rows_bf16" + variantSuffix(nativeVariant);
    }

    String profilerKey(int nativeVariant) {
        String key = activation.silu() ? "ffn_swiglu_matvec_rows_bf16" : "ffn_geglu_matvec_rows_bf16";
        return key + variantSuffix(nativeVariant);
    }

    String acceptDecision(long rows, int nativeVariant) {
        return "accept:" + activation.metalName()
                + ":native_bf16=true:native_rows=" + rows
                + ":variant=" + variantName(nativeVariant);
    }

    String capabilityRejection(DirectForwardMetalCapabilities capabilities) {
        if (activation.silu() && !capabilities.supportsSwigluFfnMatvecRowsBf16()) {
            return "swiglu_bf16_matvec_rows_symbol_unavailable";
        }
        if (activation.gelu() && !capabilities.supportsGegluFfnMatvecRowsBf16()) {
            return "geglu_bf16_matvec_rows_symbol_unavailable";
        }
        return null;
    }

    int invoke(
            MetalBinding metalBinding,
            AccelTensor output,
            DirectForwardContiguousTensor contiguousInput,
            DirectForwardMetalFfnWeightPlan weightPlan,
            DirectForwardMetalFfnShapePlan shapePlan) {
        if (activation.silu()) {
            return metalBinding.swigluFfnMatvecRowsBf16(
                    output.dataPtr(),
                    contiguousInput.tensor().dataPtr(),
                    weightPlan.gateW().dataPtr(),
                    weightPlan.upW().dataPtr(),
                    weightPlan.downW().dataPtr(),
                    Math.toIntExact(shapePlan.rows()),
                    Math.toIntExact(shapePlan.inputDim()),
                    Math.toIntExact(shapePlan.intermediateDim()),
                    Math.toIntExact(shapePlan.outputDim()));
        }
        return metalBinding.gegluFfnMatvecRowsBf16(
                output.dataPtr(),
                contiguousInput.tensor().dataPtr(),
                weightPlan.gateW().dataPtr(),
                weightPlan.upW().dataPtr(),
                weightPlan.downW().dataPtr(),
                Math.toIntExact(shapePlan.rows()),
                Math.toIntExact(shapePlan.inputDim()),
                Math.toIntExact(shapePlan.intermediateDim()),
                Math.toIntExact(shapePlan.outputDim()));
    }

    private static String variantSuffix(int nativeVariant) {
        return switch (nativeVariant) {
            case 1 -> "_scalar";
            case 2 -> "_x4";
            default -> "_unknown";
        };
    }

    private static String variantName(int nativeVariant) {
        return switch (nativeVariant) {
            case 1 -> "scalar";
            case 2 -> "x4";
            default -> "unknown";
        };
    }
}
