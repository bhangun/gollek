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

final class FlashAttentionMetalSingleLinear {
    private final MetalBinding metalBinding;
    private final FlashAttentionMetalLinearWeights weights;
    private final FlashAttentionMatvecPolicy matvecPolicy;

    FlashAttentionMetalSingleLinear(MetalBinding metalBinding, FlashAttentionMetalLinearWeights weights) {
        this(metalBinding, weights, FlashAttentionMatvecOptions.defaults());
    }

    FlashAttentionMetalSingleLinear(MetalBinding metalBinding, FlashAttentionMetalLinearWeights weights,
            FlashAttentionMatvecOptions matvecOptions) {
        this.metalBinding = metalBinding;
        this.weights = weights;
        this.matvecPolicy = FlashAttentionMatvecPolicy.from(matvecOptions);
    }

    AccelTensor tryHalfLinear(AccelTensor input, AccelTensor weight, AccelTensor bias,
            ModelConfig config, FlashAttentionModelPolicy modelPolicy, String profileKey, AccelTensor outputBuffer) {
        if (input == null || weight == null || input.quantType() != AccelTensor.QuantType.F32) {
            return null;
        }
        AccelTensor.QuantType quantType = weight.quantType();
        boolean nativeBf16Weight = weights.shouldUseNativeMetalBf16Linear(modelPolicy, weight);
        if (quantType == AccelTensor.QuantType.BF16
                && modelPolicy.disallowBf16ToF16LinearConversion()
                && !nativeBf16Weight) {
            return null;
        }
        if (quantType != AccelTensor.QuantType.F16
                && !weights.canUseMixedHalfPairWeight(weight, modelPolicy)) {
            return null;
        }
        if (!weight.isContiguous()) {
            return null;
        }
        FlashAttentionMetalLinearPlan plan = FlashAttentionMetalLinearPlan.resolve(input, weight);
        if (plan == null) {
            return null;
        }

        AccelTensor contiguousInput = input.contiguous();
        AccelTensor out = plan.reusableOutputTensor(0, outputBuffer);

        try {
            int m = plan.m();
            int kk = plan.kk();
            int n = plan.n(0);
            int rc = -2;
            String executionPath = nativeBf16Weight ? "metal_matmul_bf16" : "metal_matmul_f16";
            if (m == 1
                    && nativeBf16Weight
                    && matvecPolicy.shouldUseMetalHalfMatvec(modelPolicy, n)
                    && metalBinding.supportsMatvecTransposedRightBf16()) {
                rc = metalBinding.matvecTransposedRightBf16(
                        out.dataPtr(),
                        contiguousInput.dataPtr(),
                        weight.dataPtr(),
                        kk, n);
                if (rc == 0) {
                    executionPath = "bf16_matvec";
                }
            }
            if (rc != 0
                    && m == 1
                    && !nativeBf16Weight
                    && matvecPolicy.shouldUseMetalTransposedHalfMatvec(modelPolicy, n)
                    && metalBinding.supportsMatvecTransposedWeightHalf()) {
                AccelTensor transposedWeight = weight.toF16Transposed2dCachedUpTo(
                        FlashAttentionRuntimeOptions.metalF16WeightCacheMaxBytes());
                if (transposedWeight != null
                        && transposedWeight.size(0) == plan.k()
                        && transposedWeight.size(1) == plan.outputDim(0)) {
                    rc = metalBinding.matvecTransposedWeightHalf(
                            out.dataPtr(),
                            contiguousInput.dataPtr(),
                            transposedWeight.dataPtr(),
                            kk, n);
                    if (rc == 0) {
                        executionPath = "transposed_matvec";
                    }
                }
            }
            AccelTensor metalWeight = null;
            if (rc != 0) {
                metalWeight = weights.toMetalHalfWeight(weight, nativeBf16Weight, modelPolicy);
                if (metalWeight == null) {
                    return null;
                }
            }
            if (m == 1
                    && !nativeBf16Weight
                    && matvecPolicy.shouldUseMetalHalfMatvec(modelPolicy, n)
                    && metalBinding.supportsMatvecTransposedRightHalf()) {
                rc = metalBinding.matvecTransposedRightHalf(
                        out.dataPtr(),
                        contiguousInput.dataPtr(),
                        metalWeight.dataPtr(),
                        kk, n);
                if (rc == 0) {
                    executionPath = "matvec";
                }
            }
            if (rc != 0 && metalWeight != null) {
                rc = metalBinding.matmulTransposedRightHalf(
                        out.dataPtr(),
                        contiguousInput.dataPtr(),
                        metalWeight.dataPtr(),
                        m, kk, n,
                        1.0f, 0.0f,
                        nativeBf16Weight);
            }
            if (rc != 0) {
                throw new IllegalStateException("Metal matmulTransposedRightHalf failed with code " + rc);
            }
            DirectInferenceProfiler.recordLinearPath(profileKey, executionPath);
            return addBiasIfNeeded(out, bias);
        } catch (RuntimeException e) {
            out.close();
            return null;
        } finally {
            if (contiguousInput != input && !contiguousInput.isClosed()) {
                contiguousInput.close();
            }
        }
    }

    AccelTensor tryFloatLinear(AccelTensor input, AccelTensor weight, AccelTensor bias,
            AccelTensor outputBuffer) {
        if (input == null || weight == null) {
            return null;
        }
        if (input.quantType() != AccelTensor.QuantType.F32 || weight.quantType() != AccelTensor.QuantType.F32) {
            return null;
        }
        if (!weight.isContiguous()) {
            return null;
        }

        FlashAttentionMetalLinearPlan plan = FlashAttentionMetalLinearPlan.resolve(input, weight);
        if (plan == null) {
            return null;
        }

        AccelTensor contiguousInput = input.contiguous();
        AccelTensor out = plan.reusableOutputTensor(0, outputBuffer);

        try {
            int m = plan.m();
            int kk = plan.kk();
            int n = plan.n(0);
            int rc = metalBinding.matmulTransposedRight(
                    out.dataPtr(),
                    contiguousInput.dataPtr(),
                    weight.dataPtr(),
                    m, kk, n,
                    1.0f, 0.0f);
            if (rc != 0) {
                throw new IllegalStateException("Metal matmulTransposedRight failed with code " + rc);
            }
            return addBiasIfNeeded(out, bias);
        } catch (RuntimeException e) {
            out.close();
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
