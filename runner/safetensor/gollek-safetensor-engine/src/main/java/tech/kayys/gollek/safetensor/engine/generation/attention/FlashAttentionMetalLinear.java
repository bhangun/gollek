/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import tech.kayys.alkhawarizm.metal.binding.MetalBinding;
import tech.kayys.gollek.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.spi.model.ModelConfig;

import java.util.function.BooleanSupplier;

final class FlashAttentionMetalLinear {
    private final MetalBinding metalBinding;
    private final BooleanSupplier canUseMetal;
    private final FlashAttentionMetalMultiLinear multiLinear;
    private final FlashAttentionMetalSingleLinear singleLinear;
    private final FlashAttentionLinearOptions linearOptions;

    FlashAttentionMetalLinear(MetalBinding metalBinding, BooleanSupplier canUseMetal) {
        this(metalBinding, canUseMetal, FlashAttentionLinearOptions.defaults());
    }

    FlashAttentionMetalLinear(MetalBinding metalBinding, BooleanSupplier canUseMetal,
            FlashAttentionLinearOptions linearOptions) {
        this(metalBinding, canUseMetal, linearOptions, FlashAttentionMatvecOptions.defaults());
    }

    FlashAttentionMetalLinear(MetalBinding metalBinding, BooleanSupplier canUseMetal,
            FlashAttentionLinearOptions linearOptions, FlashAttentionMatvecOptions matvecOptions) {
        this(metalBinding, canUseMetal, new FlashAttentionMetalLinearWeights(), linearOptions, matvecOptions);
    }

    FlashAttentionMetalLinear(MetalBinding metalBinding, BooleanSupplier canUseMetal,
            FlashAttentionMetalLinearWeights weights) {
        this(metalBinding, canUseMetal, weights, FlashAttentionLinearOptions.defaults());
    }

    FlashAttentionMetalLinear(MetalBinding metalBinding, BooleanSupplier canUseMetal,
            FlashAttentionMetalLinearWeights weights, FlashAttentionLinearOptions linearOptions) {
        this(metalBinding, canUseMetal, weights, linearOptions, FlashAttentionMatvecOptions.defaults());
    }

    FlashAttentionMetalLinear(MetalBinding metalBinding, BooleanSupplier canUseMetal,
            FlashAttentionMetalLinearWeights weights, FlashAttentionLinearOptions linearOptions,
            FlashAttentionMatvecOptions matvecOptions) {
        this.metalBinding = metalBinding;
        this.canUseMetal = canUseMetal;
        this.multiLinear = new FlashAttentionMetalMultiLinear(metalBinding, weights, matvecOptions);
        this.singleLinear = new FlashAttentionMetalSingleLinear(metalBinding, weights, matvecOptions);
        this.linearOptions = linearOptions == null ? FlashAttentionLinearOptions.defaults() : linearOptions;
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
        if (!canUseExperimentalMetalLinear()) {
            return null;
        }
        return multiLinear.tryHalfLinearPairMixed(input, firstWeight, firstBias, secondWeight, secondBias,
                profileKey, config, modelPolicy, firstOutputBuffer, secondOutputBuffer);
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
        if (!canUseExperimentalMetalLinear()) {
            return null;
        }
        return multiLinear.tryHalfLinearTripleMixed(input, firstWeight, firstBias, secondWeight, secondBias,
                thirdWeight, thirdBias, profileKey, config, modelPolicy, firstOutputBuffer, secondOutputBuffer,
                thirdOutputBuffer);
    }

    AccelTensor tryHalfLinear(AccelTensor input, AccelTensor weight, AccelTensor bias,
            ModelConfig config, FlashAttentionModelPolicy modelPolicy, String profileKey, AccelTensor outputBuffer) {
        if (!canUseExperimentalMetalLinear()) {
            return null;
        }
        return singleLinear.tryHalfLinear(input, weight, bias, config, modelPolicy, profileKey, outputBuffer);
    }

    AccelTensor tryFloatLinear(AccelTensor input, AccelTensor weight, AccelTensor bias,
            AccelTensor outputBuffer) {
        if (!canUseExperimentalMetalLinear()) {
            return null;
        }
        return singleLinear.tryFloatLinear(input, weight, bias, outputBuffer);
    }

    private boolean canUseExperimentalMetalLinear() {
        return metalBinding != null
                && canUseMetal.getAsBoolean()
                && linearOptions.experimentalMetalLinearEnabled();
    }

}
