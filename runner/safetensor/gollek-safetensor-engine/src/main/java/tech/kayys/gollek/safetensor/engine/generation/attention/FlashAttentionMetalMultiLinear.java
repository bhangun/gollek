/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import tech.kayys.alkhawarizm.metal.binding.MetalBinding;
import tech.kayys.gollek.safetensor.core.tensor.AccelOps;
import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.DirectInferenceProfiler;
import tech.kayys.gollek.spi.model.ModelConfig;

final class FlashAttentionMetalMultiLinear {
    private final MetalBinding metalBinding;
    private final FlashAttentionMetalLinearWeights weights;
    private final FlashAttentionMatvecPolicy matvecPolicy;

    FlashAttentionMetalMultiLinear(MetalBinding metalBinding, FlashAttentionMetalLinearWeights weights) {
        this(metalBinding, weights, FlashAttentionMatvecOptions.defaults());
    }

    FlashAttentionMetalMultiLinear(MetalBinding metalBinding, FlashAttentionMetalLinearWeights weights,
            FlashAttentionMatvecOptions matvecOptions) {
        this.metalBinding = metalBinding;
        this.weights = weights;
        this.matvecPolicy = FlashAttentionMatvecPolicy.from(matvecOptions);
    }

    FlashAttentionProjector.LinearPair tryHalfLinearPairMixed(AccelTensor input,
            AccelTensor firstWeight,
            AccelTensor firstBias,
            AccelTensor secondWeight,
            AccelTensor secondBias,
            String profileKey,
            ModelConfig config,
            FlashAttentionModelPolicy modelPolicy,
            AccelTensor firstOutputBuffer,
            AccelTensor secondOutputBuffer) {
        if (!FlashAttentionRuntimeOptions.metalMixedHalfLinearPairEnabled()
                || !metalBinding.supportsMatmulTransposedRightHalfPairMixed()
                || input == null
                || input.quantType() != AccelTensor.QuantType.F32
                || !weights.canUseMixedHalfPairWeight(firstWeight, modelPolicy)
                || !weights.canUseMixedHalfPairWeight(secondWeight, modelPolicy)) {
            return null;
        }

        boolean nativeBf16Weights = weights.shouldUseNativeMetalBf16Linear(modelPolicy, firstWeight, secondWeight);
        AccelTensor firstMetalWeight = weights.toMetalHalfWeight(firstWeight, nativeBf16Weights, modelPolicy);
        AccelTensor secondMetalWeight = weights.toMetalHalfWeight(secondWeight, nativeBf16Weights, modelPolicy);
        if (firstMetalWeight == null || secondMetalWeight == null) {
            return null;
        }

        FlashAttentionMetalLinearPlan plan = FlashAttentionMetalLinearPlan.resolve(
                input, firstMetalWeight, secondMetalWeight);
        if (plan == null) {
            return null;
        }

        long t0 = System.nanoTime();
        AccelTensor contiguousInput = input.contiguous();
        AccelTensor first = plan.reusableOutputTensor(0, firstOutputBuffer);
        AccelTensor second = plan.reusableOutputTensor(1, secondOutputBuffer);

        try {
            int rc = metalBinding.matmulTransposedRightHalfPairMixed(
                    first.dataPtr(),
                    second.dataPtr(),
                    contiguousInput.dataPtr(),
                    firstMetalWeight.dataPtr(),
                    secondMetalWeight.dataPtr(),
                    plan.m(),
                    plan.kk(),
                    plan.n(0),
                    plan.n(1),
                    1.0f,
                    0.0f,
                    nativeBf16Weights);
            if (rc != 0) {
                throw new IllegalStateException("Metal matmulTransposedRightHalfPairMixed failed with code " + rc);
            }
            AccelTensor firstOut = addBiasIfNeeded(first, firstBias);
            AccelTensor secondOut = addBiasIfNeeded(second, secondBias);
            DirectInferenceProfiler.recordLinearPath(profileKey,
                    nativeBf16Weights ? "mixed_pair_matmul_bf16" : "mixed_pair_matmul_f16");
            DirectInferenceProfiler.recordLinearNanos(profileKey, System.nanoTime() - t0);
            return new FlashAttentionProjector.LinearPair(firstOut, secondOut);
        } catch (RuntimeException e) {
            first.close();
            second.close();
            return null;
        } finally {
            if (contiguousInput != input && !contiguousInput.isClosed()) {
                contiguousInput.close();
            }
        }
    }

    FlashAttentionProjector.LinearTriple tryHalfLinearTripleMixed(AccelTensor input,
            AccelTensor firstWeight,
            AccelTensor firstBias,
            AccelTensor secondWeight,
            AccelTensor secondBias,
            AccelTensor thirdWeight,
            AccelTensor thirdBias,
            String profileKey,
            ModelConfig config,
            FlashAttentionModelPolicy modelPolicy,
            AccelTensor firstOutputBuffer,
            AccelTensor secondOutputBuffer,
            AccelTensor thirdOutputBuffer) {
        if (FlashAttentionRuntimeOptions.disableMetalMixedHalfLinearTriple()
                || !FlashAttentionRuntimeOptions.metalMixedHalfLinearPairEnabled()
                || !metalBinding.supportsMatmulTransposedRightHalfTripleMixed()
                || input == null
                || input.quantType() != AccelTensor.QuantType.F32
                || !weights.canUseMixedHalfPairWeight(firstWeight, modelPolicy)
                || !weights.canUseMixedHalfPairWeight(secondWeight, modelPolicy)
                || !weights.canUseMixedHalfPairWeight(thirdWeight, modelPolicy)) {
            return null;
        }

        FlashAttentionMetalLinearPlan inputPlan = FlashAttentionMetalLinearPlan.resolveInput(input);
        if (inputPlan == null) {
            return null;
        }

        boolean nativeBf16Weights = weights.shouldUseNativeMetalBf16Linear(modelPolicy, firstWeight, secondWeight,
                thirdWeight);
        AccelTensor firstMetalWeight = weights.toMetalHalfWeight(firstWeight, nativeBf16Weights, modelPolicy);
        AccelTensor secondMetalWeight = weights.toMetalHalfWeight(secondWeight, nativeBf16Weights, modelPolicy);
        AccelTensor thirdMetalWeight = weights.toMetalHalfWeight(thirdWeight, nativeBf16Weights, modelPolicy);
        if (firstMetalWeight == null || secondMetalWeight == null || thirdMetalWeight == null) {
            return null;
        }

        FlashAttentionMetalLinearPlan plan = inputPlan.withWeights(
                firstMetalWeight, secondMetalWeight, thirdMetalWeight);
        if (plan == null) {
            return null;
        }

        long t0 = System.nanoTime();
        AccelTensor contiguousInput = input.contiguous();
        AccelTensor first = plan.reusableOutputTensor(0, firstOutputBuffer);
        AccelTensor second = plan.reusableOutputTensor(1, secondOutputBuffer);
        AccelTensor third = plan.reusableOutputTensor(2, thirdOutputBuffer);

        try {
            int m = plan.m();
            int kk = plan.kk();
            int n0 = plan.n(0);
            int n1 = plan.n(1);
            int n2 = plan.n(2);
            int rc = -2;
            String executionPath = nativeBf16Weights ? "mixed_triple_matmul_bf16" : "mixed_triple_matmul_f16";
            if (m == 1
                    && nativeBf16Weights
                    && matvecPolicy.shouldUseMetalHalfTripleMatvec(n0, n1, n2)
                    && metalBinding.supportsMatvecTransposedRightBf16TripleMixed()) {
                rc = metalBinding.matvecTransposedRightBf16TripleMixed(
                        first.dataPtr(),
                        second.dataPtr(),
                        third.dataPtr(),
                        contiguousInput.dataPtr(),
                        firstMetalWeight.dataPtr(),
                        secondMetalWeight.dataPtr(),
                        thirdMetalWeight.dataPtr(),
                        kk,
                        n0,
                        n1,
                        n2);
                if (rc == 0) {
                    executionPath = "mixed_triple_matvec_bf16";
                }
            }
            if (rc != 0
                    && m == 1
                    && !nativeBf16Weights
                    && matvecPolicy.shouldUseMetalHalfTripleMatvec(n0, n1, n2)
                    && metalBinding.supportsMatvecTransposedRightHalfTripleMixed()) {
                rc = metalBinding.matvecTransposedRightHalfTripleMixed(
                        first.dataPtr(),
                        second.dataPtr(),
                        third.dataPtr(),
                        contiguousInput.dataPtr(),
                        firstMetalWeight.dataPtr(),
                        secondMetalWeight.dataPtr(),
                        thirdMetalWeight.dataPtr(),
                        kk,
                        n0,
                        n1,
                        n2);
                if (rc == 0) {
                    executionPath = "mixed_triple_matvec_f16";
                }
            }
            if (rc != 0) {
                rc = metalBinding.matmulTransposedRightHalfTripleMixed(
                        first.dataPtr(),
                        second.dataPtr(),
                        third.dataPtr(),
                        contiguousInput.dataPtr(),
                        firstMetalWeight.dataPtr(),
                        secondMetalWeight.dataPtr(),
                        thirdMetalWeight.dataPtr(),
                        m,
                        kk,
                        n0,
                        n1,
                        n2,
                        1.0f,
                        0.0f,
                        nativeBf16Weights);
            }
            if (rc != 0) {
                throw new IllegalStateException("Metal matmulTransposedRightHalfTripleMixed failed with code " + rc);
            }
            AccelTensor firstOut = addBiasIfNeeded(first, firstBias);
            AccelTensor secondOut = addBiasIfNeeded(second, secondBias);
            AccelTensor thirdOut = addBiasIfNeeded(third, thirdBias);
            DirectInferenceProfiler.recordLinearPath(profileKey, executionPath);
            DirectInferenceProfiler.recordLinearNanos(profileKey, System.nanoTime() - t0);
            return new FlashAttentionProjector.LinearTriple(firstOut, secondOut, thirdOut);
        } catch (RuntimeException e) {
            first.close();
            second.close();
            third.close();
            return null;
        } finally {
            if (contiguousInput != input && !contiguousInput.isClosed()) {
                contiguousInput.close();
            }
        }
    }

    private AccelTensor addBiasIfNeeded(AccelTensor tensor, AccelTensor bias) {
        if (bias == null) {
            return tensor;
        }
        AccelTensor biased = AccelOps.add(tensor, bias);
        tensor.close();
        return biased;
    }
}
