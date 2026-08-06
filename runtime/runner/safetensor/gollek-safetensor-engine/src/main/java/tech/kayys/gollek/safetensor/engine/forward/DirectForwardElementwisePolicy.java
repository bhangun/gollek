/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

final class DirectForwardElementwisePolicy {
    private static final DirectForwardElementwiseOptions OPTIONS =
            DirectForwardElementwiseOptions.fromSystemProperties();
    private static final DirectForwardElementwiseRoutingPolicy ROUTING =
            DirectForwardElementwiseRoutingPolicy.from(OPTIONS);

    private DirectForwardElementwisePolicy() {
    }

    static boolean canUseMetalElementwise(
            ModelConfigTraits traits,
            int seqLen,
            boolean forceCpuForward,
            boolean canUseMetal,
            boolean nativeElementwiseKernelsAvailable,
            boolean nativeElementwiseFallbackAvailable) {
        return ROUTING.canUseMetalElementwise(
                traits,
                seqLen,
                forceCpuForward,
                canUseMetal,
                nativeElementwiseKernelsAvailable,
                nativeElementwiseFallbackAvailable);
    }

    static boolean canUseMetalLayerScalarScale(
            boolean useMetalElementwise,
            int seqLen,
            boolean metalBindingAvailable,
            boolean metalNativeScaleKernelAvailable) {
        return ROUTING.canUseMetalLayerScalarScale(
                useMetalElementwise,
                seqLen,
                metalBindingAvailable,
                metalNativeScaleKernelAvailable);
    }

    static boolean shouldUseMetalPostFfnNorm(ModelConfigTraits traits) {
        return ROUTING.shouldUseMetalPostFfnNorm(traits);
    }

    static boolean shouldBuildPerLayerInputs(ModelConfigTraits traits, int hiddenSizePerLayerInput) {
        return ROUTING.shouldBuildPerLayerInputs(traits, hiddenSizePerLayerInput);
    }

    static boolean shouldApplyLayerScalar(ModelConfigTraits traits) {
        return ROUTING.shouldApplyLayerScalar(traits);
    }

    static boolean isNativeBf16FfnWithPerLayerInputTarget(ModelConfigTraits traits) {
        return DirectForwardElementwiseRoutingPolicy.isNativeBf16FfnWithPerLayerInputTarget(traits);
    }

}
