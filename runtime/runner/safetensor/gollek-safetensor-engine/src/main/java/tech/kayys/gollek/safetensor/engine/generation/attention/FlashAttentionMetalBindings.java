/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import org.jboss.logging.Logger;
import tech.kayys.alkhawarizm.metal.binding.MetalBinding;
import tech.kayys.alkhawarizm.metal.binding.MetalFlashAttentionBinding;

final class FlashAttentionMetalBindings {
    private final MetalBinding metalBinding;
    private final MetalFlashAttentionBinding metalFa4;
    private final boolean metalReady;
    private final FlashAttentionBackendPolicy backendPolicy;

    FlashAttentionMetalBindings(MetalBinding metalBinding, MetalFlashAttentionBinding metalFa4,
            boolean metalReady, FlashAttentionBackendOptions backendOptions) {
        this.metalBinding = metalBinding;
        this.metalFa4 = metalFa4;
        this.metalReady = metalReady;
        this.backendPolicy = FlashAttentionBackendPolicy.from(backendOptions);
    }

    static FlashAttentionMetalBindings initialize(Logger log) {
        return initialize(log, FlashAttentionBackendOptions.fromSystemProperties());
    }

    static FlashAttentionMetalBindings initialize(Logger log, FlashAttentionBackendOptions backendOptions) {
        MetalBinding metalBinding;
        boolean metalReady;
        try {
            MetalBinding.initialize();
            metalBinding = MetalBinding.getInstance();
            metalBinding.init();
            String deviceName = metalBinding.deviceName();
            metalReady = metalBinding.isRuntimeActive()
                    && deviceName != null
                    && !deviceName.contains("CPU");
        } catch (Exception e) {
            log.warnf("FlashAttentionKernel: failed to initialize MetalBinding, forcing CPU fallback (%s)",
                    e.getMessage());
            MetalBinding.initializeFallback();
            metalBinding = MetalBinding.getInstance();
            metalReady = false;
        }

        MetalFlashAttentionBinding metalFa4;
        try {
            MetalFlashAttentionBinding.initialize();
            metalFa4 = MetalFlashAttentionBinding.getInstance();
        } catch (Exception e) {
            log.warnf(
                    "FlashAttentionKernel: failed to initialize MetalFlashAttentionBinding, forcing CPU fallback (%s)",
                    e.getMessage());
            MetalFlashAttentionBinding.initializeFallback();
            metalFa4 = MetalFlashAttentionBinding.getInstance();
        }
        return new FlashAttentionMetalBindings(metalBinding, metalFa4, metalReady, backendOptions);
    }

    MetalBinding metalBinding() {
        return metalBinding;
    }

    MetalFlashAttentionBinding metalFa4() {
        return metalFa4;
    }

    boolean canUseMetal() {
        return backendPolicy.canUseMetal(metalReady);
    }
}
