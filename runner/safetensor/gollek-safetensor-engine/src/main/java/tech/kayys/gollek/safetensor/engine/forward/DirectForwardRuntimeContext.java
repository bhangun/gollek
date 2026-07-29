/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import org.jboss.logging.Logger;
import tech.kayys.alkhawarizm.metal.binding.MetalBinding;

record DirectForwardRuntimeContext(
        Logger log,
        MetalBinding metalBinding,
        DirectForwardMetalCapabilities capabilities,
        boolean metalReady,
        boolean metalLinearEnabled) {

    boolean canUseMetal() {
        return !DirectForwardExecutionOptions.forceCpuForwardEnabled() && metalReady;
    }

    boolean canUseMetalElementwise(ModelConfigTraits traits, int seqLen) {
        boolean forceCpuForward = DirectForwardExecutionOptions.forceCpuForwardEnabled();
        return DirectForwardElementwisePolicy.canUseMetalElementwise(
                traits,
                seqLen,
                forceCpuForward,
                !forceCpuForward && metalReady,
                metalBinding != null && capabilities.nativeElementwiseKernelsAvailable(),
                metalBinding != null && capabilities.nativeElementwiseFallbackAvailable());
    }

    boolean canUseMetalLayerScalarScale(boolean useMetalElementwise, int seqLen) {
        return DirectForwardElementwisePolicy.canUseMetalLayerScalarScale(
                useMetalElementwise,
                seqLen,
                metalBinding != null,
                capabilities.nativeScaleKernelAvailable());
    }
}
