/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import java.util.Objects;

import tech.kayys.alkhawarizm.metal.binding.MetalBinding;
import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;

record DirectForwardMetalMatvecGatedFfnKernelPlan(
        DirectForwardMetalFfnActivationPlan activation,
        boolean nativeBf16Weights) {

    DirectForwardMetalMatvecGatedFfnKernelPlan {
        Objects.requireNonNull(activation, "activation");
        if (!activation.supported()) {
            throw new IllegalArgumentException(
                    "Unsupported matvec gated FFN activation: " + activation.activationType());
        }
    }

    static DirectForwardMetalMatvecGatedFfnKernelPlan from(
            DirectForwardMetalFfnActivationPlan activation,
            boolean nativeBf16Weights) {
        return new DirectForwardMetalMatvecGatedFfnKernelPlan(activation, nativeBf16Weights);
    }

    String pathSuffix() {
        return activation.matvecPathSuffix(nativeBf16Weights);
    }

    String profilerKey() {
        return activation.matvecProfilerKey(nativeBf16Weights);
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
        if (nativeBf16Weights && activation.silu()) {
            return metalBinding.swigluFfnMatvecBf16(
                    output.dataPtr(),
                    contiguousInput.tensor().dataPtr(),
                    weightPlan.gateW().dataPtr(),
                    weightPlan.upW().dataPtr(),
                    weightPlan.downW().dataPtr(),
                    Math.toIntExact(shapePlan.inputDim()),
                    Math.toIntExact(shapePlan.intermediateDim()),
                    Math.toIntExact(shapePlan.outputDim()));
        }
        if (nativeBf16Weights) {
            return metalBinding.gegluFfnMatvecBf16(
                    output.dataPtr(),
                    contiguousInput.tensor().dataPtr(),
                    weightPlan.gateW().dataPtr(),
                    weightPlan.upW().dataPtr(),
                    weightPlan.downW().dataPtr(),
                    Math.toIntExact(shapePlan.inputDim()),
                    Math.toIntExact(shapePlan.intermediateDim()),
                    Math.toIntExact(shapePlan.outputDim()));
        }
        if (activation.silu()) {
            return metalBinding.swigluFfnMatvecHalf(
                    output.dataPtr(),
                    contiguousInput.tensor().dataPtr(),
                    weightPlan.gateW().dataPtr(),
                    weightPlan.upW().dataPtr(),
                    weightPlan.downW().dataPtr(),
                    Math.toIntExact(shapePlan.inputDim()),
                    Math.toIntExact(shapePlan.intermediateDim()),
                    Math.toIntExact(shapePlan.outputDim()));
        }
        return metalBinding.gegluFfnMatvecHalf(
                output.dataPtr(),
                contiguousInput.tensor().dataPtr(),
                weightPlan.gateW().dataPtr(),
                weightPlan.upW().dataPtr(),
                weightPlan.downW().dataPtr(),
                Math.toIntExact(shapePlan.inputDim()),
                Math.toIntExact(shapePlan.intermediateDim()),
                Math.toIntExact(shapePlan.outputDim()));
    }
}
