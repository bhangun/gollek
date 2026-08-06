/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import static tech.kayys.gollek.safetensor.engine.forward.DirectForwardTensorOps.addBiasIfNeeded;
import static tech.kayys.gollek.safetensor.engine.forward.DirectForwardTensorOps.reusableOutputTensor;

import org.jboss.logging.Logger;
import tech.kayys.alkhawarizm.metal.binding.MetalBinding;
import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.DirectInferenceProfiler;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;

final class DirectForwardMetalHalfLinearPair {
    private static final String PATH = "gate-up-pair";
    static final String PROFILE_KEY = "ffn_gate_up_pair";

    private DirectForwardMetalHalfLinearPair() {
    }

    static Result tryPair(Logger log,
            MetalBinding metalBinding,
            DirectForwardMetalCapabilities capabilities,
            ModelConfigTraits traits,
            ModelConfig config,
            boolean metalLinearEnabled,
            boolean decodeLogitsPhase,
            AccelTensor input,
            AccelTensor firstWeight,
            AccelTensor firstBias,
            AccelTensor secondWeight,
            AccelTensor secondBias,
            AccelTensor firstOutputBuffer,
            AccelTensor secondOutputBuffer) {
        DirectForwardMetalHalfLinearPairAdmissionPlan admissionPlan = DirectForwardMetalHalfLinearPairAdmissionPlan
                .from(
                        capabilities,
                        traits,
                        metalLinearEnabled,
                        input,
                        firstWeight,
                        secondWeight);
        if (!admissionPlan.admitted()) {
            trace(admissionPlan.rejectionDecision(), config, input, firstWeight, secondWeight);
            return null;
        }
        DirectForwardMetalLinearShapePlan shapePlan = DirectForwardMetalLinearShapePlan.pair(input, firstWeight,
                secondWeight);
        if (shapePlan == null) {
            trace("reject:shape_mismatch", config, input, firstWeight, secondWeight);
            return null;
        }

        DirectForwardMetalLinearWeightPlan weightPlan = DirectForwardMetalLinearWeightPlan.pair(
                traits,
                PROFILE_KEY,
                decodeLogitsPhase,
                shapePlan.rows(),
                firstWeight,
                secondWeight);
        if (!weightPlan.hasUniformPairHalfOrBf16Weights()) {
            trace("reject:" + weightPlan.conversionFailureReason(), config, input, firstWeight, secondWeight);
            return null;
        }
        boolean nativeBf16Weights = weightPlan.nativeBf16Weights();
        AccelTensor metalFirstWeight = weightPlan.firstWeight();
        AccelTensor metalSecondWeight = weightPlan.secondWeight();

        long t0 = System.nanoTime();
        AccelTensor first = reusableOutputTensor(firstOutputBuffer, shapePlan.outputShape());
        AccelTensor second = reusableOutputTensor(secondOutputBuffer, shapePlan.outputShape());

        try (DirectForwardContiguousTensor contiguousInput = DirectForwardContiguousTensor.from(input)) {
            int m = Math.toIntExact(shapePlan.rows());
            int kk = Math.toIntExact(shapePlan.inputDim());
            int n = Math.toIntExact(shapePlan.outputDim());
            DirectForwardMetalHalfLinearPairExecutionPlan executionPlan = DirectForwardMetalHalfLinearPairExecutionPlan
                    .from(
                            m,
                            kk,
                            n,
                            nativeBf16Weights,
                            capabilities,
                            DirectForwardMetalHalfMatvecPolicy.shouldUseMetalHalfMatvecPair(traits, config, n));
            int rc = -2;
            String executionPath = executionPlan.matmulPath();
            if (executionPlan.nativeBf16MatvecCandidate()) {
                rc = metalBinding.matvecTransposedRightBf16Pair(
                        first.dataPtr(),
                        second.dataPtr(),
                        contiguousInput.tensor().dataPtr(),
                        metalFirstWeight.dataPtr(),
                        metalSecondWeight.dataPtr(),
                        kk, n);
                if (rc == 0) {
                    executionPath = executionPlan.nativeBf16MatvecPath();
                }
            }
            if (rc != 0 && executionPlan.halfMatvecCandidate()) {
                rc = metalBinding.matvecTransposedRightHalfPair(
                        first.dataPtr(),
                        second.dataPtr(),
                        contiguousInput.tensor().dataPtr(),
                        metalFirstWeight.dataPtr(),
                        metalSecondWeight.dataPtr(),
                        kk, n);
                if (rc == 0) {
                    executionPath = executionPlan.halfMatvecPath();
                }
            }
            if (rc != 0) {
                rc = metalBinding.matmulTransposedRightHalfPair(
                        first.dataPtr(),
                        second.dataPtr(),
                        contiguousInput.tensor().dataPtr(),
                        metalFirstWeight.dataPtr(),
                        metalSecondWeight.dataPtr(),
                        m, kk, n,
                        1.0f, 0.0f,
                        executionPlan.nativeBf16Weights());
            }
            if (rc != 0) {
                throw new IllegalStateException("Metal matmulTransposedRightHalfPair failed with code " + rc);
            }
            AccelTensor firstOut = addBiasIfNeeded(first, firstBias);
            AccelTensor secondOut = addBiasIfNeeded(second, secondBias);
            DirectInferenceProfiler.recordLinearPath(PROFILE_KEY, executionPath);
            DirectInferenceProfiler.recordLinearNanos(PROFILE_KEY, System.nanoTime() - t0);
            trace("accept:" + executionPath + ":native_bf16=" + nativeBf16Weights,
                    config, input, firstWeight, secondWeight);
            return new Result(firstOut, secondOut);
        } catch (RuntimeException e) {
            first.close();
            second.close();
            trace("reject:runtime_failure:" + e.getClass().getSimpleName(),
                    config, input, firstWeight, secondWeight);
            log.debugf("Falling back from Metal half linear pair to separate linears: %s", e.getMessage());
            return null;
        }
    }

    private static void trace(String decision,
            ModelConfig config,
            AccelTensor input,
            AccelTensor firstWeight,
            AccelTensor secondWeight) {
        DirectForwardFfnFastPathTrace.trace(PATH, decision, config, input, firstWeight, secondWeight, null);
    }

    record Result(AccelTensor first, AccelTensor second) {
    }
}
