/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import tech.kayys.alkhawarizm.metal.binding.MetalBinding;
import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.alkhawarizm.spi.model.FFNActivationType;

record DirectForwardMetalGateUpMatvecFfnAdmissionPlan(
        DirectForwardMetalFfnActivationPlan activation,
        String rejectionReason) {

    static DirectForwardMetalGateUpMatvecFfnAdmissionPlan from(
            MetalBinding metalBinding,
            boolean metalLinearEnabled,
            FFNActivationType activationType,
            AccelTensor gateB,
            AccelTensor upB,
            AccelTensor combinedBuffer) {
        return from(
                metalBinding != null,
                metalLinearEnabled,
                activationType,
                gateB,
                upB,
                combinedBuffer);
    }

    static DirectForwardMetalGateUpMatvecFfnAdmissionPlan from(
            boolean metalBindingAvailable,
            boolean metalLinearEnabled,
            FFNActivationType activationType,
            AccelTensor gateB,
            AccelTensor upB,
            AccelTensor combinedBuffer) {
        DirectForwardMetalFfnActivationPlan activation = DirectForwardMetalFfnActivationPlan.from(activationType);
        if (!DirectForwardFfnFastPathPolicy.shouldUseMetalGateUpMatvecFfn()) {
            return reject(activation, "flag_disabled");
        }
        if (!activation.supported()) {
            return reject(activation, activation.unsupportedReason());
        }
        if (gateB != null || upB != null) {
            return reject(activation, "bias_present");
        }
        if (combinedBuffer == null || combinedBuffer.isClosed()) {
            return reject(activation, "combined_workspace_unavailable");
        }
        if (!metalLinearEnabled || !metalBindingAvailable) {
            return reject(activation, "metal_unavailable");
        }
        return new DirectForwardMetalGateUpMatvecFfnAdmissionPlan(activation, null);
    }

    boolean admitted() {
        return rejectionReason == null;
    }

    String rejectionDecision() {
        return admitted() ? null : "reject:" + rejectionReason;
    }

    private static DirectForwardMetalGateUpMatvecFfnAdmissionPlan reject(
            DirectForwardMetalFfnActivationPlan activation,
            String reason) {
        return new DirectForwardMetalGateUpMatvecFfnAdmissionPlan(activation, reason);
    }
}
