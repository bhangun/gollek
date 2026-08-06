/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;

record DirectForwardMetalFfnCandidatePlan(String rejectionReason) {

    static DirectForwardMetalFfnCandidatePlan fused(
            boolean metalLinearEnabled,
            AccelTensor input,
            AccelTensor gateW,
            AccelTensor upW,
            AccelTensor downW,
            ModelConfigTraits traits,
            String profileKey) {
        if (!canUseProjection(metalLinearEnabled, input, gateW, traits, profileKey)) {
            return reject("gate_candidate_ineligible");
        }
        if (!canUseProjection(metalLinearEnabled, input, upW, traits, profileKey)) {
            return reject("up_candidate_ineligible");
        }
        if (!canUseOutputWeight(downW, traits, profileKey)) {
            return reject("down_weight_ineligible");
        }
        return admit();
    }

    static DirectForwardMetalFfnCandidatePlan matvecGated(
            boolean metalLinearEnabled,
            AccelTensor input,
            AccelTensor gateW,
            AccelTensor upW,
            AccelTensor downW,
            ModelConfigTraits traits,
            String profileKey) {
        if (!canUseProjection(metalLinearEnabled, input, gateW, traits, profileKey)
                || !canUseProjection(metalLinearEnabled, input, upW, traits, profileKey)
                || !canUseOutputWeight(downW, traits, profileKey)) {
            return reject("candidate_ineligible");
        }
        return admit();
    }

    static DirectForwardMetalFfnCandidatePlan gateUpMatvec(
            boolean metalLinearEnabled,
            AccelTensor input,
            AccelTensor gateW,
            AccelTensor upW,
            ModelConfigTraits traits,
            String profileKey) {
        if (!canUseProjection(metalLinearEnabled, input, gateW, traits, profileKey)
                || !canUseProjection(metalLinearEnabled, input, upW, traits, profileKey)) {
            return reject("candidate_ineligible");
        }
        return admit();
    }

    boolean admitted() {
        return rejectionReason == null;
    }

    String rejectionDecision() {
        return admitted() ? null : "reject:" + rejectionReason;
    }

    private static boolean canUseProjection(
            boolean metalLinearEnabled,
            AccelTensor input,
            AccelTensor weight,
            ModelConfigTraits traits,
            String profileKey) {
        return DirectForwardMetalLinearPolicy.canUseMetalHalfLinearCandidate(
                metalLinearEnabled, input, weight, traits, profileKey);
    }

    private static boolean canUseOutputWeight(
            AccelTensor weight,
            ModelConfigTraits traits,
            String profileKey) {
        return DirectForwardMetalLinearPolicy.canUseMetalHalfWeight(weight, traits, profileKey);
    }

    private static DirectForwardMetalFfnCandidatePlan admit() {
        return new DirectForwardMetalFfnCandidatePlan(null);
    }

    private static DirectForwardMetalFfnCandidatePlan reject(String reason) {
        return new DirectForwardMetalFfnCandidatePlan(reason);
    }
}
