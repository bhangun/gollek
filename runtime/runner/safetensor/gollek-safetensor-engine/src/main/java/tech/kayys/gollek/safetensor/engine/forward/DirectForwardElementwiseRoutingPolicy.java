/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import java.util.Objects;

record DirectForwardElementwiseRoutingPolicy(DirectForwardElementwiseOptions options) {

    DirectForwardElementwiseRoutingPolicy {
        options = Objects.requireNonNull(options, "options");
    }

    static DirectForwardElementwiseRoutingPolicy from(DirectForwardElementwiseOptions options) {
        return new DirectForwardElementwiseRoutingPolicy(options);
    }

    boolean canUseMetalElementwise(
            ModelConfigTraits traits,
            int seqLen,
            boolean forceCpuForward,
            boolean canUseMetal,
            boolean nativeElementwiseKernelsAvailable,
            boolean nativeElementwiseFallbackAvailable) {
        if (forceCpuForward) {
            return false;
        }
        boolean nativeBf16 = traits.nativeBf16Matvec();
        if (nativeBf16) {
            if (options.disableNativeBf16MetalElementwise()) {
                return false;
            }
            if (!canUseMetal || !nativeElementwiseKernelsAvailable) {
                return false;
            }
        } else if (!canUseMetal && !nativeElementwiseFallbackAvailable) {
            return false;
        }
        int defaultMinSeq = nativeBf16 ? 1 : 16;
        int minSeq = options.metalElementwiseMinSeq() >= 0
                ? options.metalElementwiseMinSeq()
                : defaultMinSeq;
        if (seqLen < minSeq) {
            return false;
        }
        if (!nativeBf16) {
            return true;
        }
        return options.enableNativeBf16MetalElementwise() || nativeElementwiseKernelsAvailable;
    }

    boolean canUseMetalLayerScalarScale(
            boolean useMetalElementwise,
            int seqLen,
            boolean metalBindingAvailable,
            boolean metalNativeScaleKernelAvailable) {
        return useMetalElementwise
                && metalBindingAvailable
                && metalNativeScaleKernelAvailable
                && (seqLen >= options.metalLayerScalarMinSeq()
                || (seqLen == 1 && options.enableMetalLayerScalarDecode()));
    }

    boolean shouldUseMetalPostFfnNorm(ModelConfigTraits traits) {
        if (options.disableMetalPostFfnNorm()) {
            return false;
        }
        if (options.enableMetalPostFfnNorm() != null) {
            return options.enableMetalPostFfnNorm();
        }
        return isNativeBf16FfnWithPerLayerInputTarget(traits);
    }

    boolean shouldBuildPerLayerInputs(ModelConfigTraits traits, int hiddenSizePerLayerInput) {
        if (hiddenSizePerLayerInput <= 0) {
            return false;
        }
        return !isPerLayerInputEmbeddingDisabled(traits);
    }

    boolean shouldApplyLayerScalar(ModelConfigTraits traits) {
        if (!traits.nativeBf16Matvec()) {
            return true;
        }
        return !options.disableNativeBf16LayerScalar();
    }

    static boolean isNativeBf16FfnWithPerLayerInputTarget(ModelConfigTraits traits) {
        return traits.nativeBf16Matvec() || traits.perLayerInputEmbedding();
    }

    private boolean isPerLayerInputEmbeddingDisabled(ModelConfigTraits traits) {
        return traits.perLayerInputEmbedding() && options.disablePerLayerInputEmbedding();
    }
}
