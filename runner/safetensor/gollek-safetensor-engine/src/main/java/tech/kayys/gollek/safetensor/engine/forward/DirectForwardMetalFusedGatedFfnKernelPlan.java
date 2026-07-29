/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import java.util.Objects;

import tech.kayys.alkhawarizm.metal.binding.MetalBinding;
import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;

record DirectForwardMetalFusedGatedFfnKernelPlan(
        DirectForwardMetalFfnActivationPlan activation,
        boolean nativeBf16Weights) {

    DirectForwardMetalFusedGatedFfnKernelPlan {
        Objects.requireNonNull(activation, "activation");
        if (!activation.supported()) {
            throw new IllegalArgumentException(
                    "Unsupported fused gated FFN activation: " + activation.activationType());
        }
    }

    static DirectForwardMetalFusedGatedFfnKernelPlan from(
            DirectForwardMetalFfnActivationPlan activation,
            boolean nativeBf16Weights) {
        return new DirectForwardMetalFusedGatedFfnKernelPlan(activation, nativeBf16Weights);
    }

    String pathSuffix() {
        return activation.fusedPathSuffix(nativeBf16Weights);
    }

    String profilerKey() {
        return activation.fusedProfilerKey();
    }

    String acceptDecision() {
        return activation.acceptDecision(nativeBf16Weights);
    }

    int invoke(
            MetalBinding metalBinding,
            AccelTensor output,
            DirectForwardContiguousTensor contiguousInput,
            DirectForwardMetalFfnWeightPlan weightPlan,
            DirectForwardMetalFfnShapePlan shapePlan) {
        if (activation.silu()) {
            return metalBinding.swigluFfnHalf(
                    output.dataPtr(),
                    contiguousInput.tensor().dataPtr(),
                    weightPlan.gateW().dataPtr(),
                    weightPlan.upW().dataPtr(),
                    weightPlan.downW().dataPtr(),
                    Math.toIntExact(shapePlan.rows()),
                    Math.toIntExact(shapePlan.inputDim()),
                    Math.toIntExact(shapePlan.intermediateDim()),
                    Math.toIntExact(shapePlan.outputDim()),
                    nativeBf16Weights);
        }
        return metalBinding.gegluFfnHalf(
                output.dataPtr(),
                contiguousInput.tensor().dataPtr(),
                weightPlan.gateW().dataPtr(),
                weightPlan.upW().dataPtr(),
                weightPlan.downW().dataPtr(),
                Math.toIntExact(shapePlan.rows()),
                Math.toIntExact(shapePlan.inputDim()),
                Math.toIntExact(shapePlan.intermediateDim()),
                Math.toIntExact(shapePlan.outputDim()),
                nativeBf16Weights);
    }
}
