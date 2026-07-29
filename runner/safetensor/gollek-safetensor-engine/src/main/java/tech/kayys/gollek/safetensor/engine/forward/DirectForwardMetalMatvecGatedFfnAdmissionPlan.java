/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import tech.kayys.alkhawarizm.metal.binding.MetalBinding;
import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.spi.model.FFNActivationType;

record DirectForwardMetalMatvecGatedFfnAdmissionPlan(
        DirectForwardMetalFfnActivationPlan activation,
        String rejectionReason) {

    static DirectForwardMetalMatvecGatedFfnAdmissionPlan from(
            MetalBinding metalBinding,
            ModelConfigTraits traits,
            boolean metalLinearEnabled,
            FFNActivationType activationType,
            AccelTensor gateB,
            AccelTensor upB,
            AccelTensor downB) {
        return from(
                metalBinding != null,
                traits,
                metalLinearEnabled,
                activationType,
                gateB,
                upB,
                downB);
    }

    static DirectForwardMetalMatvecGatedFfnAdmissionPlan from(
            boolean metalBindingAvailable,
            ModelConfigTraits traits,
            boolean metalLinearEnabled,
            FFNActivationType activationType,
            AccelTensor gateB,
            AccelTensor upB,
            AccelTensor downB) {
        DirectForwardMetalFfnActivationPlan activation = DirectForwardMetalFfnActivationPlan.from(activationType);
        String flagRejection = matvecFlagRejection(activation, traits);
        if (flagRejection != null) {
            return reject(activation, flagRejection);
        }
        if (!activation.supported()) {
            return reject(activation, activation.unsupportedReason());
        }
        if (gateB != null || upB != null || downB != null) {
            return reject(activation, "bias_present");
        }
        if (!metalLinearEnabled || !metalBindingAvailable) {
            return reject(activation, "metal_unavailable");
        }
        return new DirectForwardMetalMatvecGatedFfnAdmissionPlan(activation, null);
    }

    boolean admitted() {
        return rejectionReason == null;
    }

    String rejectionDecision() {
        return admitted() ? null : "reject:" + rejectionReason;
    }

    private static String matvecFlagRejection(
            DirectForwardMetalFfnActivationPlan activation,
            ModelConfigTraits traits) {
        if (activation.gelu() && !DirectForwardFfnFastPathPolicy.shouldUseMetalGegluMatvecFfn(traits)) {
            return "geglu_flag_disabled";
        }
        if (activation.silu() && !DirectForwardFfnFastPathPolicy.shouldUseMetalSwigluMatvecFfn(traits)) {
            return "swiglu_flag_disabled";
        }
        return null;
    }

    private static DirectForwardMetalMatvecGatedFfnAdmissionPlan reject(
            DirectForwardMetalFfnActivationPlan activation,
            String reason) {
        return new DirectForwardMetalMatvecGatedFfnAdmissionPlan(activation, reason);
    }
}
