/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import static tech.kayys.gollek.safetensor.engine.forward.DirectForwardTensorOps.addBiasIfNeeded;
import static tech.kayys.gollek.safetensor.engine.forward.DirectForwardTensorOps.reusableOutputTensor;

import java.lang.foreign.MemorySegment;
import org.jboss.logging.Logger;
import tech.kayys.alkhawarizm.metal.binding.MetalBinding;
import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.DirectInferenceProfiler;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;

final class DirectForwardMetalHalfLinear {
    private DirectForwardMetalHalfLinear() {
    }

    static AccelTensor tryLinear(Logger log,
            MetalBinding metalBinding,
            DirectForwardMetalCapabilities capabilities,
            ModelConfigTraits traits,
            ModelConfig config,
            boolean metalLinearEnabled,
            boolean decodeLogitsPhase,
            AccelTensor input,
            AccelTensor weight,
            AccelTensor bias,
            AccelTensor outputBuffer,
            String profileKey) {
        DirectForwardMetalHalfLinearAdmissionPlan admissionPlan = DirectForwardMetalHalfLinearAdmissionPlan.from(
                traits,
                metalLinearEnabled,
                input,
                weight,
                profileKey);
        if (!admissionPlan.admitted()) {
            return null;
        }
        DirectForwardMetalLinearShapePlan shapePlan = DirectForwardMetalLinearShapePlan.single(input, weight);
        if (shapePlan == null) {
            return null;
        }
        DirectForwardMetalLinearWeightPlan weightPlan = DirectForwardMetalLinearWeightPlan.single(
                traits,
                profileKey,
                decodeLogitsPhase,
                shapePlan.rows(),
                weight);
        AccelTensor out = reusableOutputTensor(outputBuffer, shapePlan.outputShape());

        try (DirectForwardContiguousTensor contiguousInput = DirectForwardContiguousTensor.from(input)) {
            int m = Math.toIntExact(shapePlan.rows());
            int kk = Math.toIntExact(shapePlan.inputDim());
            int n = Math.toIntExact(shapePlan.outputDim());
            boolean nf4Weight = weightPlan.weight().quantType() == AccelTensor.QuantType.NF4;
            boolean int4Weight = weightPlan.weight().quantType() == AccelTensor.QuantType.INT4;
            DirectForwardMetalHalfLinearExecutionPlan executionPlan = DirectForwardMetalHalfLinearExecutionPlan.from(
                    m,
                    kk,
                    n,
                    weightPlan.nativeBf16Weights(),
                    nf4Weight,
                    int4Weight,
                    capabilities,
                    DirectForwardMetalHalfMatvecPolicy.shouldUseMetalHalfMatvec(
                            traits, config, n, profileKey),
                    DirectForwardMetalHalfMatvecPolicy.shouldUseMetalLogitsMpsMatvec(
                            traits, n, kk, profileKey),
                    DirectForwardMetalHalfMatvecPolicy.shouldUseMetalTransposedHalfMatvec(
                            traits, n, profileKey));
            int rc = -2;
            String executionPath = executionPlan.matmulPath();
            AccelTensor metalWeight = null;
            if (rc != 0 && executionPlan.nf4MatvecCandidate()) {
                MemorySegment packed = weight.dataPtr();
                MemorySegment absmax = weight.scales();
                if (packed != null && absmax != null) {
                    if (m == 1) {
                        try (DirectForwardContiguousTensor f32Input = DirectForwardContiguousTensor
                                .from(input.dequantizeTransient())) {
                            MemorySegment outPtr = out.dataPtr();
                            MemorySegment inPtr = f32Input.tensor().dataPtr();
                            long outStride = (long) n * Float.BYTES;
                            long inStride = (long) kk * Float.BYTES;
                            rc = 0;
                            for (int i = 0; i < m; i++) {
                                int rowRc = metalBinding.matvecTransposedRightNf4(
                                        outPtr.asSlice(i * outStride),
                                        inPtr.asSlice(i * inStride),
                                        packed,
                                        absmax,
                                        kk, n, weight.groupSize());
                                if (rowRc != 0) {
                                    rc = rowRc;
                                    break;
                                }
                            }
                        }
                    } else {
                        rc = -2;
                    }
                    if (rc == 0) {
                        executionPath = executionPlan.nf4MatvecPath();
                    }
                }
            }
            if (rc != 0 && executionPlan.int4MatvecCandidate()) {
                metalWeight = weightPlan.weight();
                if (metalWeight != null) {
                    if (m == 1) {
                        try (DirectForwardContiguousTensor f32Input = DirectForwardContiguousTensor
                                .from(input.dequantizeTransient())) {
                            MemorySegment outPtr = out.dataPtr();
                            MemorySegment inPtr = f32Input.tensor().dataPtr();
                            MemorySegment wPtr = metalWeight.dataPtr();
                            long outStride = (long) n * Float.BYTES;
                            long inStride = (long) kk * Float.BYTES;
                            rc = 0;
                            for (int i = 0; i < m; i++) {
                                int rowRc = metalBinding.matvecTransposedRightInt4(
                                        outPtr.asSlice(i * outStride),
                                        inPtr.asSlice(i * inStride),
                                        wPtr,
                                        metalWeight.scales(),
                                        kk, n, weight.groupSize());
                                if (rowRc != 0) {
                                    rc = rowRc;
                                    break;
                                }
                            }
                        }
                    } else {
                        rc = -2;
                    }
                    if (rc == 0) {
                        executionPath = executionPlan.int4MatvecPath();
                    }
                }
            }
            if (rc != 0 && executionPlan.nativeBf16MatvecCandidate()) {
                metalWeight = DirectForwardMetalLinearWeightPlan.toMetalHalfWeight(
                        weight, traits, true, weightPlan.allowBf16ToF16Weights());
                if (metalWeight != null) {
                    MemorySegment outPtr = out.dataPtr();
                    MemorySegment inPtr = contiguousInput.tensor().dataPtr();
                    MemorySegment wPtr = metalWeight.dataPtr();
                    long outStride = (long) n * Float.BYTES;
                    long inStride = (long) kk * Float.BYTES;
                    rc = 0;
                    for (int i = 0; i < m; i++) {
                        int rowRc = metalBinding.matvecTransposedRightBf16(
                                outPtr.asSlice(i * outStride),
                                inPtr.asSlice(i * inStride),
                                wPtr,
                                kk, n);
                        if (rowRc != 0) {
                            rc = rowRc;
                            break;
                        }
                    }
                    if (rc == 0) {
                        executionPath = executionPlan.nativeBf16MatvecPath();
                    }
                }
            }
            if (rc != 0 && executionPlan.mpsHalfMatvecCandidate()) {
                metalWeight = DirectForwardMetalLinearWeightPlan.toMetalHalfWeight(
                        weight, traits, false, weightPlan.allowBf16ToF16Weights());
                if (metalWeight != null) {
                    rc = metalBinding.matvecTransposedRightHalfMps(
                            out.dataPtr(),
                            contiguousInput.tensor().dataPtr(),
                            metalWeight.dataPtr(),
                            kk, n);
                    if (rc == 0) {
                        executionPath = executionPlan.mpsHalfMatvecPath();
                    }
                }
            }
            if (rc != 0 && executionPlan.transposedHalfMatvecCandidate()) {
                AccelTensor transposedWeight = DirectForwardLinearCachePolicy.cachedTransposedF16Weight(weight);
                if (executionPlan.matchesTransposedWeight(transposedWeight)) {
                    rc = metalBinding.matvecTransposedWeightHalf(
                            out.dataPtr(),
                            contiguousInput.tensor().dataPtr(),
                            transposedWeight.dataPtr(),
                            kk, n);
                    if (rc == 0) {
                        executionPath = executionPlan.transposedHalfMatvecPath();
                    }
                }
            }
            if (rc != 0 && executionPlan.halfMatvecCandidate()) {
                if (metalWeight == null) {
                    metalWeight = DirectForwardMetalLinearWeightPlan.toMetalHalfWeight(
                            weight, traits, false, weightPlan.allowBf16ToF16Weights());
                }
                if (metalWeight != null) {
                    rc = metalBinding.matvecTransposedRightHalf(
                            out.dataPtr(),
                            contiguousInput.tensor().dataPtr(),
                            metalWeight.dataPtr(),
                            kk, n);
                    if (rc == 0) {
                        executionPath = executionPlan.halfMatvecPath();
                    }
                }
            }
            if (rc != 0) {
                if (metalWeight == null) {
                    metalWeight = weightPlan.weight();
                }
                if (metalWeight == null) {
                    out.close();
                    return null;
                }
                if (executionPlan.nf4Weight()) {
                    MemorySegment packed = weight.dataPtr();
                    MemorySegment absmax = weight.scales();
                    if (packed != null && absmax != null) {
                        for (int i = 0; i < m; i++) {
                            rc = metalBinding.matvecTransposedRightNf4(
                                    out.dataPtr().asSlice((long) i * n * 4L),
                                    contiguousInput.tensor().dataPtr().asSlice((long) i * kk * 4L),
                                    packed,
                                    absmax,
                                    kk, n, weight.groupSize());
                            if (rc != 0)
                                break;
                        }
                        if (rc == 0) {
                            executionPath = executionPlan.nf4MatvecPath(); // Or a new path for matmul
                        }
                    }
                } else {
                    rc = metalBinding.matmulTransposedRightHalf(
                            out.dataPtr(),
                            contiguousInput.tensor().dataPtr(),
                            metalWeight.dataPtr(),
                            m, kk, n,
                            1.0f, 0.0f,
                            executionPlan.nativeBf16Weight());
                }
            }
            if (rc != 0) {
                log.infof("Metal linear failed with rc=%d, m=%d, halfMatvecCandidate=%b, nativeBf16Weight=%b", rc, m,
                        executionPlan.halfMatvecCandidate(), executionPlan.nativeBf16Weight());
                throw new IllegalStateException("Metal matmulTransposedRightHalf failed with code " + rc);
            }
            DirectInferenceProfiler.recordLinearPath(profileKey, executionPath);
            return addBiasIfNeeded(out, bias);
        } catch (RuntimeException e) {
            out.close();
            System.err.println("Falling back from Metal half linear to AccelOps: " + e.getMessage());
            return null;
        }
    }
}
