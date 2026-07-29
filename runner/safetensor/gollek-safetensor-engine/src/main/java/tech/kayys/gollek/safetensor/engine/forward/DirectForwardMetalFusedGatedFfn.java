/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import static tech.kayys.gollek.safetensor.engine.forward.DirectForwardTensorOps.reusableOutputTensor;

import org.jboss.logging.Logger;
import tech.kayys.alkhawarizm.metal.binding.MetalBinding;
import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.DirectInferenceProfiler;
import tech.kayys.gollek.spi.model.FFNActivationType;
import tech.kayys.gollek.spi.model.ModelConfig;

final class DirectForwardMetalFusedGatedFfn {
    private static final String PATH = "fused-gated-ffn";
    private static final String PROFILE_KEY = "ffn_fused_metal";

    private DirectForwardMetalFusedGatedFfn() {
    }

    static String skipReason(MetalBinding metalBinding,
            DirectForwardMetalCapabilities capabilities,
            ModelConfigTraits traits,
            boolean metalLinearEnabled,
            AccelTensor input,
            FFNActivationType activationType,
            AccelTensor gateB,
            AccelTensor upB,
            AccelTensor downB) {
        return DirectForwardMetalFusedGatedFfnAdmissionPlan.from(
                metalBinding,
                capabilities,
                traits,
                metalLinearEnabled,
                input,
                activationType,
                gateB,
                upB,
                downB).rejectionReason();
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
            AccelTensor downW,
            AccelTensor downB,
            AccelTensor outputBuffer) {
        DirectForwardMetalFusedGatedFfnAdmissionPlan admissionPlan = DirectForwardMetalFusedGatedFfnAdmissionPlan.from(
                metalBinding,
                capabilities,
                traits,
                metalLinearEnabled,
                input,
                activationType,
                gateB,
                upB,
                downB);
        if (!admissionPlan.admitted()) {
            return reject(admissionPlan.rejectionReason(), config, input, gateW, upW, downW);
        }
        DirectForwardMetalFfnActivationPlan activation = admissionPlan.activation();
        long rows = admissionPlan.rows();
        DirectForwardMetalFfnCandidatePlan candidatePlan = DirectForwardMetalFfnCandidatePlan.fused(
                metalLinearEnabled, input, gateW, upW, downW, traits, PROFILE_KEY);
        if (!candidatePlan.admitted()) {
            return reject(candidatePlan.rejectionReason(), config, input, gateW, upW, downW);
        }

        DirectForwardMetalFfnShapeAdmissionPlan shapeAdmission = DirectForwardMetalFfnShapeAdmissionPlan.gated(input,
                gateW, upW, downW);
        if (!shapeAdmission.admitted()) {
            return reject(shapeAdmission.rejectionReason(), config, input, gateW, upW, downW);
        }
        DirectForwardMetalFfnShapePlan shapePlan = shapeAdmission.shapePlan();

        DirectForwardMetalFfnWeightPlan weightPlan = DirectForwardMetalFfnWeightPlan.gated(
                traits,
                PROFILE_KEY,
                decodeLogitsPhase,
                rows,
                gateW,
                upW,
                downW);
        String weightRejection = weightPlan.gatedConversionFailureReason();
        if (weightRejection != null) {
            return reject(weightRejection, config, input, gateW, upW, downW);
        }
        boolean nativeBf16Weights = weightPlan.nativeBf16Weights();
        DirectForwardMetalFusedGatedFfnKernelPlan kernelPlan = DirectForwardMetalFusedGatedFfnKernelPlan
                .from(activation, nativeBf16Weights);

        long t0 = System.nanoTime();
        AccelTensor out = reusableOutputTensor(outputBuffer, shapePlan.outputShape());

        try (DirectForwardContiguousTensor contiguousInput = DirectForwardContiguousTensor.from(input)) {
            int rc = kernelPlan.invoke(metalBinding, out, contiguousInput, weightPlan, shapePlan);
            if (rc != 0) {
                throw new IllegalStateException("Metal fused gated FFN failed with code " + rc);
            }
            DirectInferenceProfiler.recordLinearPath(PROFILE_KEY, kernelPlan.pathSuffix());
            DirectInferenceProfiler.recordLinearNanos(kernelPlan.profilerKey(), System.nanoTime() - t0);
            trace(kernelPlan.acceptDecision(), config, input, gateW, upW, downW);
            return out;
        } catch (RuntimeException e) {
            out.close();
            trace("reject:runtime_failure:" + e.getClass().getSimpleName(), config, input, gateW, upW, downW);
            log.debugf("Falling back from fused Metal gated FFN: %s", e.getMessage());
            return null;
        }
    }

    private static AccelTensor reject(String reason,
            ModelConfig config,
            AccelTensor input,
            AccelTensor gateW,
            AccelTensor upW,
            AccelTensor downW) {
        trace("reject:" + reason, config, input, gateW, upW, downW);
        return null;
    }

    private static void trace(String decision,
            ModelConfig config,
            AccelTensor input,
            AccelTensor gateW,
            AccelTensor upW,
            AccelTensor downW) {
        DirectForwardFfnFastPathTrace.trace(PATH, decision, config, input, gateW, upW, downW);
    }
}
