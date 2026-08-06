/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

record FlashAttentionBackendOptions(boolean forceCpuForward) {
    private static final String FORCE_CPU_FORWARD_PROPERTY = "gollek.safetensor.force_cpu_forward";

    static FlashAttentionBackendOptions fromSystemProperties() {
        return new FlashAttentionBackendOptions(Boolean.getBoolean(FORCE_CPU_FORWARD_PROPERTY));
    }

    static FlashAttentionBackendOptions defaults() {
        return new FlashAttentionBackendOptions(false);
    }
}
