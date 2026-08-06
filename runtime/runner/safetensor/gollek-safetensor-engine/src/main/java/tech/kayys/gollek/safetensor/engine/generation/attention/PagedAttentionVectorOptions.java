/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

record PagedAttentionVectorOptions(boolean debugAttentionProbe) {
    private static final String DEBUG_ATTENTION_PROBE_PROPERTY = "gollek.debug.attention_probe";

    static PagedAttentionVectorOptions fromSystemProperties() {
        return new PagedAttentionVectorOptions(Boolean.getBoolean(DEBUG_ATTENTION_PROBE_PROPERTY));
    }

    static PagedAttentionVectorOptions defaults() {
        return new PagedAttentionVectorOptions(false);
    }
}
