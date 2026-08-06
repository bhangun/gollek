/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;

record DirectForwardMetalHalfLinearAdmissionPlan(
        boolean admitted,
        String rejectionDecision) {

    static DirectForwardMetalHalfLinearAdmissionPlan from(
            ModelConfigTraits traits,
            boolean metalLinearEnabled,
            AccelTensor input,
            AccelTensor weight,
            String profileKey) {
        if (!DirectForwardMetalLinearPolicy.canUseMetalHalfLinearCandidate(
                metalLinearEnabled, input, weight, traits, profileKey)) {
            return reject("candidate_ineligible");
        }
        return new DirectForwardMetalHalfLinearAdmissionPlan(true, null);
    }

    private static DirectForwardMetalHalfLinearAdmissionPlan reject(String reason) {
        return new DirectForwardMetalHalfLinearAdmissionPlan(false, "reject:" + reason);
    }
}
