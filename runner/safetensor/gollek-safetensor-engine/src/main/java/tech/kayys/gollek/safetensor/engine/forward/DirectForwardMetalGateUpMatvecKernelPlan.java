/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import java.util.Objects;

import tech.kayys.alkhawarizm.metal.binding.MetalBinding;
import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;

record DirectForwardMetalGateUpMatvecKernelPlan(
        DirectForwardMetalFfnActivationPlan activation,
        boolean nativeBf16Weights) {

    DirectForwardMetalGateUpMatvecKernelPlan {
        Objects.requireNonNull(activation, "activation");
        if (!activation.supported()) {
            throw new IllegalArgumentException("Unsupported gate/up matvec activation: " + activation.activationType());
        }
    }

    static DirectForwardMetalGateUpMatvecKernelPlan from(
            DirectForwardMetalFfnActivationPlan activation,
            boolean nativeBf16Weights) {
        return new DirectForwardMetalGateUpMatvecKernelPlan(activation, nativeBf16Weights);
    }

    String pathSuffix() {
        return activation.gateUpPathSuffix(nativeBf16Weights);
    }

    String acceptDecision() {
        return activation.acceptDecision(nativeBf16Weights);
    }

    int invoke(
            MetalBinding metalBinding,
            AccelTensor combinedBuffer,
            DirectForwardContiguousTensor contiguousInput,
            DirectForwardMetalFfnWeightPlan weightPlan,
            DirectForwardMetalFfnShapePlan shapePlan) {
        if (nativeBf16Weights && activation.silu()) {
            return metalBinding.swigluGateUpMatvecBf16(
                    combinedBuffer.dataPtr(),
                    contiguousInput.tensor().dataPtr(),
                    weightPlan.gateW().dataPtr(),
                    weightPlan.upW().dataPtr(),
                    Math.toIntExact(shapePlan.inputDim()),
                    Math.toIntExact(shapePlan.intermediateDim()));
        }
        if (nativeBf16Weights) {
            return metalBinding.gegluGateUpMatvecBf16(
                    combinedBuffer.dataPtr(),
                    contiguousInput.tensor().dataPtr(),
                    weightPlan.gateW().dataPtr(),
                    weightPlan.upW().dataPtr(),
                    Math.toIntExact(shapePlan.inputDim()),
                    Math.toIntExact(shapePlan.intermediateDim()));
        }
        if (activation.silu()) {
            return metalBinding.swigluGateUpMatvecHalf(
                    combinedBuffer.dataPtr(),
                    contiguousInput.tensor().dataPtr(),
                    weightPlan.gateW().dataPtr(),
                    weightPlan.upW().dataPtr(),
                    Math.toIntExact(shapePlan.inputDim()),
                    Math.toIntExact(shapePlan.intermediateDim()));
        }
        return metalBinding.gegluGateUpMatvecHalf(
                combinedBuffer.dataPtr(),
                contiguousInput.tensor().dataPtr(),
                weightPlan.gateW().dataPtr(),
                weightPlan.upW().dataPtr(),
                Math.toIntExact(shapePlan.inputDim()),
                Math.toIntExact(shapePlan.intermediateDim()));
    }
}
