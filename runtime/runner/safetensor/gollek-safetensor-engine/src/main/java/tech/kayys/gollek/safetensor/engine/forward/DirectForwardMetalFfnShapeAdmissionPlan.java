/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;

record DirectForwardMetalFfnShapeAdmissionPlan(
        DirectForwardMetalFfnShapePlan shapePlan,
        String rejectionReason) {

    static DirectForwardMetalFfnShapeAdmissionPlan gated(
            AccelTensor input,
            AccelTensor gateW,
            AccelTensor upW,
            AccelTensor downW) {
        return admitShape(DirectForwardMetalFfnShapePlan.gated(input, gateW, upW, downW));
    }

    static DirectForwardMetalFfnShapeAdmissionPlan singleTokenGated(
            AccelTensor input,
            AccelTensor gateW,
            AccelTensor upW,
            AccelTensor downW) {
        return requireSingleToken(DirectForwardMetalFfnShapePlan.gated(input, gateW, upW, downW));
    }

    static DirectForwardMetalFfnShapeAdmissionPlan singleTokenGateUp(
            AccelTensor input,
            AccelTensor gateW,
            AccelTensor upW,
            AccelTensor combinedBuffer) {
        DirectForwardMetalFfnShapeAdmissionPlan admission =
                requireSingleToken(DirectForwardMetalFfnShapePlan.gateUp(input, gateW, upW));
        if (!admission.admitted()) {
            return admission;
        }
        if (!admission.shapePlan().matchesOutputBuffer(combinedBuffer)) {
            return reject(admission.shapePlan(), "combined_shape_mismatch");
        }
        return admission;
    }

    boolean admitted() {
        return rejectionReason == null;
    }

    String rejectionDecision() {
        return admitted() ? null : "reject:" + rejectionReason;
    }

    private static DirectForwardMetalFfnShapeAdmissionPlan admitShape(
            DirectForwardMetalFfnShapePlan shapePlan) {
        return shapePlan == null
                ? reject("shape_mismatch")
                : new DirectForwardMetalFfnShapeAdmissionPlan(shapePlan, null);
    }

    private static DirectForwardMetalFfnShapeAdmissionPlan requireSingleToken(
            DirectForwardMetalFfnShapePlan shapePlan) {
        DirectForwardMetalFfnShapeAdmissionPlan admission = admitShape(shapePlan);
        if (!admission.admitted() || shapePlan.singleRow()) {
            return admission;
        }
        return reject(shapePlan, "not_single_token_rows:" + shapePlan.rows());
    }

    private static DirectForwardMetalFfnShapeAdmissionPlan reject(String reason) {
        return reject(null, reason);
    }

    private static DirectForwardMetalFfnShapeAdmissionPlan reject(
            DirectForwardMetalFfnShapePlan shapePlan,
            String reason) {
        return new DirectForwardMetalFfnShapeAdmissionPlan(shapePlan, reason);
    }
}
