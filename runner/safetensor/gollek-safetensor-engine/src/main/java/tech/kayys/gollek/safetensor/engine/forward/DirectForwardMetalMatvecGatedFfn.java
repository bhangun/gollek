/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import static tech.kayys.gollek.safetensor.engine.forward.DirectForwardTensorOps.allFinite;
import static tech.kayys.gollek.safetensor.engine.forward.DirectForwardTensorOps.isSingleRowLinearInput;
import static tech.kayys.gollek.safetensor.engine.forward.DirectForwardTensorOps.reusableOutputTensor;

import org.jboss.logging.Logger;
import tech.kayys.alkhawarizm.metal.binding.MetalBinding;
import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.DirectInferenceProfiler;
import tech.kayys.gollek.spi.model.FFNActivationType;
import tech.kayys.gollek.spi.model.ModelConfig;

final class DirectForwardMetalMatvecGatedFfn {
    private static final String PATH = "matvec-gated-ffn";
    private static final String PROFILE_KEY = "ffn_matvec_gated";

    private DirectForwardMetalMatvecGatedFfn() {
    }

    static boolean shouldAttempt(AccelTensor input, ModelConfigTraits traits, FFNActivationType activationType) {
        if (!isSingleRowLinearInput(input)) {
            return false;
        }
        return DirectForwardMetalMatvecGatedFfnAdmissionPlan.from(
                true,
                traits,
                true,
                activationType,
                null,
                null,
                null).admitted();
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
        DirectForwardMetalMatvecGatedFfnAdmissionPlan admissionPlan = DirectForwardMetalMatvecGatedFfnAdmissionPlan
                .from(
                        metalBinding,
                        traits,
                        metalLinearEnabled,
                        activationType,
                        gateB,
                        upB,
                        downB);
        if (!admissionPlan.admitted()) {
            trace(admissionPlan.rejectionDecision(), config, input, gateW, upW, downW);
            return null;
        }
        DirectForwardMetalFfnActivationPlan activation = admissionPlan.activation();
        DirectForwardMetalFfnCandidatePlan candidatePlan = DirectForwardMetalFfnCandidatePlan.matvecGated(
                metalLinearEnabled, input, gateW, upW, downW, traits, PROFILE_KEY);
        if (!candidatePlan.admitted()) {
            trace(candidatePlan.rejectionDecision(), config, input, gateW, upW, downW);
            return null;
        }

        DirectForwardMetalFfnShapeAdmissionPlan shapeAdmission = DirectForwardMetalFfnShapeAdmissionPlan
                .singleTokenGated(input, gateW, upW, downW);
        if (!shapeAdmission.admitted()) {
            trace(shapeAdmission.rejectionDecision(), config, input, gateW, upW, downW);
            return null;
        }
        DirectForwardMetalFfnShapePlan shapePlan = shapeAdmission.shapePlan();

        boolean allowBf16ToF16Weights = DirectForwardMetalFfnWeightPlan.allowBf16ToF16Weights(
                shapePlan.rows(),
                traits,
                PROFILE_KEY,
                decodeLogitsPhase);
        boolean nativeBf16Weights = DirectForwardMetalFfnWeightPlan.nativeBf16Weights(
                traits, PROFILE_KEY, allowBf16ToF16Weights, gateW, upW, downW);
        String capabilityRejection = activation.matvecCapabilityRejection(capabilities, nativeBf16Weights);
        if (capabilityRejection != null) {
            trace("reject:" + capabilityRejection, config, input, gateW, upW, downW);
            return null;
        }

        DirectForwardMetalFfnWeightPlan weightPlan = DirectForwardMetalFfnWeightPlan.gated(
                traits, gateW, upW, downW, nativeBf16Weights, allowBf16ToF16Weights);
        String conversionRejection = weightPlan.gatedConversionFailureReason();
        if (conversionRejection != null) {
            trace("reject:" + conversionRejection, config, input, gateW, upW, downW);
            return null;
        }
        String typeRejection = weightPlan.gatedUniformTypeFailureReason();
        if (typeRejection != null) {
            trace("reject:" + typeRejection, config, input, gateW, upW, downW);
            return null;
        }
        DirectForwardMetalMatvecGatedFfnKernelPlan kernelPlan = DirectForwardMetalMatvecGatedFfnKernelPlan
                .from(activation, nativeBf16Weights);

        long t0 = System.nanoTime();
        AccelTensor out = reusableOutputTensor(outputBuffer, shapePlan.outputShape());

        try (DirectForwardContiguousTensor contiguousInput = DirectForwardContiguousTensor.from(input)) {
            int rc = kernelPlan.invoke(metalBinding, out, contiguousInput, weightPlan, shapePlan);
            if (rc != 0) {
                throw new IllegalStateException("Metal matvec gated FFN failed with code " + rc);
            }
            if (DirectForwardFfnFastPathPolicy.shouldValidateMetalMatvecFfn(
                    DirectForwardFfnFastPathTrace.isEnabled()) && !allFinite(out)) {
                throw new IllegalStateException("Metal matvec gated FFN produced non-finite output");
            }
            DirectInferenceProfiler.recordLinearPath(PROFILE_KEY, kernelPlan.pathSuffix());
            DirectInferenceProfiler.recordLinearNanos(kernelPlan.profilerKey(), System.nanoTime() - t0);
            trace(kernelPlan.acceptDecision(), config, input, gateW, upW, downW);
            return out;
        } catch (RuntimeException e) {
            out.close();
            trace("reject:runtime_failure:" + e.getClass().getSimpleName(), config, input, gateW, upW, downW);
            System.err.println("Falling back from Metal matvec gated FFN: " + e.getMessage());
            return null;
        }
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
