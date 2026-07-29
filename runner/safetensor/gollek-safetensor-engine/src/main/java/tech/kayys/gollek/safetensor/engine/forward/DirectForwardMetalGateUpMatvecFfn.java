/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import org.jboss.logging.Logger;
import tech.kayys.alkhawarizm.metal.binding.MetalBinding;
import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.DirectInferenceProfiler;
import tech.kayys.gollek.spi.model.FFNActivationType;
import tech.kayys.gollek.spi.model.ModelConfig;

final class DirectForwardMetalGateUpMatvecFfn {
    private static final String PATH = "gate-up-gated-matvec";
    private static final String PROFILE_KEY = "ffn_gate_up_gated";

    private DirectForwardMetalGateUpMatvecFfn() {
    }

    static AccelTensor tryFfn(Logger log,
            MetalBinding metalBinding,
            DirectForwardMetalCapabilities capabilities,
            ModelConfigTraits traits,
            ModelConfig config,
            boolean metalLinearEnabled,
            boolean decodeLogitsPhase,
            AccelTensor input,
            FFNActivationType activationType,
            AccelTensor gateW,
            AccelTensor gateB,
            AccelTensor upW,
            AccelTensor upB,
            AccelTensor combinedBuffer) {
        DirectForwardMetalGateUpMatvecFfnAdmissionPlan admissionPlan = DirectForwardMetalGateUpMatvecFfnAdmissionPlan
                .from(
                        metalBinding,
                        metalLinearEnabled,
                        activationType,
                        gateB,
                        upB,
                        combinedBuffer);
        if (!admissionPlan.admitted()) {
            trace(admissionPlan.rejectionDecision(), config, input, gateW, upW);
            return null;
        }
        DirectForwardMetalFfnActivationPlan activation = admissionPlan.activation();
        DirectForwardMetalFfnCandidatePlan candidatePlan = DirectForwardMetalFfnCandidatePlan.gateUpMatvec(
                metalLinearEnabled, input, gateW, upW, traits, PROFILE_KEY);
        if (!candidatePlan.admitted()) {
            trace(candidatePlan.rejectionDecision(), config, input, gateW, upW);
            return null;
        }
        DirectForwardMetalFfnShapeAdmissionPlan shapeAdmission = DirectForwardMetalFfnShapeAdmissionPlan
                .singleTokenGateUp(input, gateW, upW, combinedBuffer);
        if (!shapeAdmission.admitted()) {
            trace(shapeAdmission.rejectionDecision(), config, input, gateW, upW);
            return null;
        }
        DirectForwardMetalFfnShapePlan shapePlan = shapeAdmission.shapePlan();

        boolean allowBf16ToF16Weights = DirectForwardMetalFfnWeightPlan.allowBf16ToF16Weights(
                shapePlan.rows(),
                traits,
                PROFILE_KEY,
                decodeLogitsPhase);
        boolean nativeBf16Weights = DirectForwardMetalFfnWeightPlan.nativeBf16Weights(
                traits, PROFILE_KEY, allowBf16ToF16Weights, gateW, upW);
        String capabilityRejection = activation.gateUpCapabilityRejection(capabilities, nativeBf16Weights);
        if (capabilityRejection != null) {
            trace("reject:" + capabilityRejection, config, input, gateW, upW);
            return null;
        }
        DirectForwardMetalFfnWeightPlan weightPlan = DirectForwardMetalFfnWeightPlan.gateUp(
                traits, gateW, upW, nativeBf16Weights, allowBf16ToF16Weights);
        String weightRejection = weightPlan.gateUpConversionFailureReason();
        if (weightRejection != null) {
            trace("reject:" + weightRejection, config, input, gateW, upW);
            return null;
        }

        DirectForwardMetalGateUpMatvecKernelPlan kernelPlan = DirectForwardMetalGateUpMatvecKernelPlan.from(activation,
                nativeBf16Weights);

        long t0 = System.nanoTime();
        try (DirectForwardContiguousTensor contiguousInput = DirectForwardContiguousTensor.from(input)) {
            int rc = kernelPlan.invoke(metalBinding, combinedBuffer, contiguousInput, weightPlan, shapePlan);
            if (rc != 0) {
                throw new IllegalStateException("Metal gated gate/up matvec failed with code " + rc);
            }
            DirectInferenceProfiler.recordLinearPath(PROFILE_KEY, kernelPlan.pathSuffix());
            DirectInferenceProfiler.recordLinearNanos(PROFILE_KEY, System.nanoTime() - t0);
            trace(kernelPlan.acceptDecision(), config, input, gateW, upW);
            return combinedBuffer;
        } catch (RuntimeException e) {
            trace("reject:runtime_failure:" + e.getClass().getSimpleName(), config, input, gateW, upW);
            log.debugf("Falling back from Metal gated gate/up matvec: %s", e.getMessage());
            return null;
        }
    }

    private static void trace(String decision,
            ModelConfig config,
            AccelTensor input,
            AccelTensor gateW,
            AccelTensor upW) {
        DirectForwardFfnFastPathTrace.trace(PATH, decision, config, input, gateW, upW, null);
    }
}
