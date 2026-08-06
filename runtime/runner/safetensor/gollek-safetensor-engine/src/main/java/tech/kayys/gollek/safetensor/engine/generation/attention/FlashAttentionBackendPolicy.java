/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

record FlashAttentionBackendPolicy(FlashAttentionBackendOptions options) {

    FlashAttentionBackendPolicy {
        if (options == null) {
            options = FlashAttentionBackendOptions.defaults();
        }
    }

    static FlashAttentionBackendPolicy from(FlashAttentionBackendOptions options) {
        return new FlashAttentionBackendPolicy(options);
    }

    boolean canUseMetal(boolean metalReady) {
        return metalReady && !options.forceCpuForward();
    }
}
