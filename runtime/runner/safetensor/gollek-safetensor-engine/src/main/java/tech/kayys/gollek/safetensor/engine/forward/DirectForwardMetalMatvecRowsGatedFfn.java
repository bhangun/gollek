/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import static tech.kayys.gollek.safetensor.engine.forward.DirectForwardTensorOps.allFinite;
import static tech.kayys.gollek.safetensor.engine.forward.DirectForwardTensorOps.isPackedF32LinearInput;
import static tech.kayys.gollek.safetensor.engine.forward.DirectForwardTensorOps.linearRowView;
import static tech.kayys.gollek.safetensor.engine.forward.DirectForwardTensorOps.reusableOutputTensor;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.DirectInferenceProfiler;

/**
 * Experimental short-prefill adapter that reuses the single-row Metal gated FFN
 * matvec kernel across prompt rows, preferring the native row-batched BF16
 * kernel when the local Metal bridge exposes it. It is opt-in because the
 * production fused GEGLU prefill path is still better for most multi-row
 * batches.
 */
final class DirectForwardMetalMatvecRowsGatedFfn {
    private static final String PATH = "matvec-gated-ffn-prefill-rows";
    private static final String PROFILE_KEY = "ffn_matvec_gated_rows";

    private DirectForwardMetalMatvecRowsGatedFfn() {
    }

    static boolean shouldAttempt(DirectForwardGatedFfnRequest request) {
        if (!DirectForwardFfnFastPathPolicy.shouldUseMetalMatvecFfnPrefillRows(request.traits(), 2L)) {
            return false;
        }
        DirectForwardMetalFfnShapeAdmissionPlan shapeAdmission = shapeAdmission(request);
        return shapeAdmission.admitted()
                && request.runtime().metalBinding() != null
                && isPackedF32LinearInput(request.input())
                && DirectForwardFfnFastPathPolicy.shouldUseMetalMatvecFfnPrefillRows(
                        request.traits(), shapeAdmission.shapePlan().rows());
    }

    static boolean shouldDeferToFusedPrefill(DirectForwardGatedFfnRequest request) {
        DirectForwardMetalFfnShapeAdmissionPlan shapeAdmission = shapeAdmission(request);
        return shapeAdmission.admitted()
                && DirectForwardFfnFastPathPolicy.shouldPreferMetalFusedFfnPrefillOverMatvecRows(
                        request.traits(), shapeAdmission.shapePlan().rows());
    }

    static void recordStrategySkip(DirectForwardGatedFfnRequest request, String reason) {
        trace(request, "skip:" + reason);
    }

    static AccelTensor tryFfn(DirectForwardGatedFfnRequest request) {
        DirectForwardMetalFfnShapeAdmissionPlan shapeAdmission = shapeAdmission(request);
        if (!shapeAdmission.admitted()) {
            return reject(request, shapeAdmission.rejectionDecision());
        }
        DirectForwardMetalFfnShapePlan shapePlan = shapeAdmission.shapePlan();
        long rows = shapePlan.rows();
        String preflightRejection = preflightRejection(request, rows);
        if (preflightRejection != null) {
            return reject(request, preflightRejection);
        }

        AccelTensor output = reusableOutputTensor(request.downOutputBuffer(), shapePlan.outputShape());
        boolean callerOwnedOutput = output == request.downOutputBuffer();
        try {
            if (tryNativeRows(request, output, shapePlan)) {
                return output;
            }
            for (long row = 0L; row < rows; row++) {
                if (!runRow(request, output, row)) {
                    closeIfOwned(output, callerOwnedOutput);
                    return reject(request, "reject:row_failed:" + row);
                }
            }
            trace(request, "accept:rows=" + rows);
            return output;
        } catch (RuntimeException e) {
            closeIfOwned(output, callerOwnedOutput);
            trace(request, "reject:runtime_failure:" + e.getClass().getSimpleName());
            request.runtime().log().debugf("Falling back from row-prefill Metal matvec gated FFN: %s", e.getMessage());
            return null;
        }
    }

    private static boolean tryNativeRows(
            DirectForwardGatedFfnRequest request,
            AccelTensor output,
            DirectForwardMetalFfnShapePlan shapePlan) {
        DirectForwardMetalFfnActivationPlan activation =
                DirectForwardMetalFfnActivationPlan.from(request.activationType());
        if (!activation.supported()) {
            trace(request, "reject:native_rows:" + activation.unsupportedReason());
            return false;
        }
        DirectForwardMetalFfnCandidatePlan candidatePlan = DirectForwardMetalFfnCandidatePlan.matvecGated(
                request.metalLinearEnabled(),
                request.input(),
                request.gateW(),
                request.upW(),
                request.downW(),
                request.traits(),
                PROFILE_KEY);
        if (!candidatePlan.admitted()) {
            trace(request, "reject:native_rows:" + candidatePlan.rejectionReason());
            return false;
        }
        DirectForwardMetalMatvecRowsGatedFfnKernelPlan kernelPlan =
                DirectForwardMetalMatvecRowsGatedFfnKernelPlan.from(activation);
        String capabilityRejection = kernelPlan.capabilityRejection(request.runtime().capabilities());
        if (capabilityRejection != null) {
            trace(request, "reject:native_rows:" + capabilityRejection);
            return false;
        }

        boolean allowBf16ToF16Weights = DirectForwardMetalFfnWeightPlan.allowBf16ToF16Weights(
                shapePlan.rows(), request.traits(), PROFILE_KEY, request.decodeLogitsPhase());
        boolean nativeBf16Weights = DirectForwardMetalFfnWeightPlan.nativeBf16Weights(
                request.traits(), PROFILE_KEY, allowBf16ToF16Weights,
                request.gateW(), request.upW(), request.downW());
        if (!nativeBf16Weights) {
            trace(request, "reject:native_rows:native_bf16_required");
            return false;
        }

        DirectForwardMetalFfnWeightPlan weightPlan = DirectForwardMetalFfnWeightPlan.gated(
                request.traits(), request.gateW(), request.upW(), request.downW(), true, false);
        String conversionRejection = weightPlan.gatedConversionFailureReason();
        if (conversionRejection != null) {
            trace(request, "reject:native_rows:" + conversionRejection);
            return false;
        }
        String typeRejection = weightPlan.gatedUniformTypeFailureReason();
        if (typeRejection != null) {
            trace(request, "reject:native_rows:" + typeRejection);
            return false;
        }

        long t0 = System.nanoTime();
        try (DirectForwardContiguousTensor contiguousInput = DirectForwardContiguousTensor.from(request.input())) {
            int rc = kernelPlan.invoke(request.runtime().metalBinding(), output, contiguousInput, weightPlan, shapePlan);
            if (rc != 0) {
                trace(request, "reject:native_rows:rc=" + rc);
                return false;
            }
            if (DirectForwardFfnFastPathPolicy.shouldValidateMetalMatvecFfn(
                    DirectForwardFfnFastPathTrace.isEnabled()) && !allFinite(output)) {
                trace(request, "reject:native_rows:non_finite_output");
                return false;
            }
            int nativeVariant = request.runtime().metalBinding().bf16FfnMatvecRowsVariant();
            DirectInferenceProfiler.recordLinearPath(PROFILE_KEY, kernelPlan.pathSuffix(nativeVariant));
            DirectInferenceProfiler.recordLinearNanos(kernelPlan.profilerKey(nativeVariant), System.nanoTime() - t0);
            trace(request, kernelPlan.acceptDecision(shapePlan.rows(), nativeVariant));
            return true;
        } catch (RuntimeException e) {
            trace(request, "reject:native_rows:runtime_failure:" + e.getClass().getSimpleName());
            request.runtime().log().debugf(
                    "Falling back from native row-batched Metal matvec gated FFN: %s", e.getMessage());
            return false;
        }
    }

    private static boolean runRow(DirectForwardGatedFfnRequest request, AccelTensor output, long row) {
        AccelTensor inputRow = linearRowView(request.input(), row);
        AccelTensor outputRow = linearRowView(output, row);
        AccelTensor rowResult = null;
        try {
            rowResult = DirectForwardMetalMatvecGatedFfn.tryFfn(
                    request.runtime().log(),
                    request.runtime().metalBinding(),
                    request.runtime().capabilities(),
                    request.traits(),
                    request.config(),
                    request.metalLinearEnabled(),
                    request.decodeLogitsPhase(),
                    inputRow,
                    request.activationType(),
                    request.gateW(),
                    request.gateB(),
                    request.upW(),
                    request.upB(),
                    request.downW(),
                    request.downB(),
                    outputRow);
            return rowResult == outputRow;
        } finally {
            if (rowResult != null && rowResult != outputRow && !rowResult.isClosed()) {
                rowResult.close();
            }
            if (!outputRow.isClosed()) {
                outputRow.close();
            }
            if (!inputRow.isClosed()) {
                inputRow.close();
            }
        }
    }

    private static String preflightRejection(DirectForwardGatedFfnRequest request, long rows) {
        if (!isPackedF32LinearInput(request.input())) {
            return "reject:input_not_packed_f32";
        }
        if (!DirectForwardFfnFastPathPolicy.shouldUseMetalMatvecFfnPrefillRows(request.traits(), rows)) {
            return "reject:flag_or_row_limit:rows=" + rows;
        }
        AccelTensor firstRow = linearRowView(request.input(), 0L);
        try {
            if (!DirectForwardMetalMatvecGatedFfn.shouldAttempt(
                    firstRow, request.traits(), request.activationType())) {
                return "reject:single_row_gate";
            }
        } finally {
            firstRow.close();
        }
        return null;
    }

    private static DirectForwardMetalFfnShapeAdmissionPlan shapeAdmission(DirectForwardGatedFfnRequest request) {
        return DirectForwardMetalFfnShapeAdmissionPlan.gated(
                request.input(), request.gateW(), request.upW(), request.downW());
    }

    private static AccelTensor reject(DirectForwardGatedFfnRequest request, String decision) {
        trace(request, decision);
        return null;
    }

    private static void closeIfOwned(AccelTensor output, boolean callerOwnedOutput) {
        if (!callerOwnedOutput && output != null && !output.isClosed()) {
            output.close();
        }
    }

    private static void trace(DirectForwardGatedFfnRequest request, String decision) {
        DirectForwardFfnFastPathTrace.trace(
                PATH,
                decision,
                request.config(),
                request.input(),
                request.gateW(),
                request.upW(),
                request.downW());
    }
}
