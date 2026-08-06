/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import static tech.kayys.gollek.safetensor.engine.forward.DirectForwardTensorOps.isMultiRowLinearInput;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;

record DirectForwardMetalHalfLinearPairAdmissionPlan(
        boolean admitted,
        String rejectionDecision) {

    static DirectForwardMetalHalfLinearPairAdmissionPlan from(
            DirectForwardMetalCapabilities capabilities,
            ModelConfigTraits traits,
            boolean metalLinearEnabled,
            AccelTensor input,
            AccelTensor firstWeight,
            AccelTensor secondWeight) {
        return from(
                capabilities,
                traits,
                metalLinearEnabled,
                input,
                firstWeight,
                secondWeight,
                DirectForwardMetalHalfMatvecPolicy.shouldUseMetalHalfLinearPair(
                        traits,
                        isMultiRowLinearInput(input),
                        DirectForwardFfnFastPathPolicy.isNativeBf16FusedHalfFfnAllowed()));
    }

    static DirectForwardMetalHalfLinearPairAdmissionPlan from(
            DirectForwardMetalCapabilities capabilities,
            ModelConfigTraits traits,
            boolean metalLinearEnabled,
            AccelTensor input,
            AccelTensor firstWeight,
            AccelTensor secondWeight,
            boolean pairFastPathAllowed) {
        if (!pairFastPathAllowed) {
            return reject("disabled");
        }
        if (!DirectForwardMetalLinearPolicy.canUseMetalHalfLinearCandidate(
                metalLinearEnabled, input, firstWeight, traits, DirectForwardMetalHalfLinearPair.PROFILE_KEY)) {
            return reject("first_candidate_ineligible");
        }
        if (!DirectForwardMetalLinearPolicy.canUseMetalHalfLinearCandidate(
                metalLinearEnabled, input, secondWeight, traits, DirectForwardMetalHalfLinearPair.PROFILE_KEY)) {
            return reject("second_candidate_ineligible");
        }
        if (!capabilities.supportsMatmulTransposedRightHalfPair()) {
            return reject("pair_symbol_unavailable");
        }
        return new DirectForwardMetalHalfLinearPairAdmissionPlan(true, null);
    }

    private static DirectForwardMetalHalfLinearPairAdmissionPlan reject(String reason) {
        return new DirectForwardMetalHalfLinearPairAdmissionPlan(false, "reject:" + reason);
    }
}
