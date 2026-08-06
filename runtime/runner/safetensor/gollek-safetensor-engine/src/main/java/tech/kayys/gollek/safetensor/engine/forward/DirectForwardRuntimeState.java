/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import jakarta.annotation.PostConstruct;
import jakarta.inject.Singleton;
import org.jboss.logging.Logger;
import tech.kayys.alkhawarizm.metal.binding.MetalBinding;

/**
 * Owns platform runtime initialization and capability detection for direct
 * forward execution.
 */
@Singleton
public class DirectForwardRuntimeState {
    private MetalBinding metalBinding;
    private boolean metalReady;
    private DirectForwardMetalCapabilities metalCapabilities = DirectForwardMetalCapabilities.EMPTY;

    @PostConstruct
    void init() {
        try {
            MetalBinding.initialize();
            this.metalBinding = MetalBinding.getInstance();
            this.metalBinding.init();
            String deviceName = this.metalBinding.deviceName();
            this.metalReady = this.metalBinding.isRuntimeActive()
                    && deviceName != null
                    && !deviceName.contains("CPU");
            refreshMetalCapabilities();
        } catch (Exception e) {
            MetalBinding.initializeFallback();
            this.metalBinding = MetalBinding.getInstance();
            this.metalReady = false;
            refreshMetalCapabilities();
        }
    }

    DirectForwardRuntimeContext context(Logger log) {
        boolean metalLinearEnabled = !DirectForwardExecutionOptions.forceCpuForwardEnabled()
                && metalReady
                && DirectForwardMetalLinearPolicy.experimentalMetalLinearEnabled();
        return new DirectForwardRuntimeContext(
                log,
                metalBinding,
                metalCapabilities,
                metalReady,
                metalLinearEnabled);
    }

    private void refreshMetalCapabilities() {
        this.metalCapabilities = DirectForwardMetalCapabilities.detect(this.metalBinding);
    }
}
