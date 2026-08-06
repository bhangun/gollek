/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import static tech.kayys.gollek.safetensor.engine.forward.DirectForwardRuntimeOptions.parseOptionalBoolean;

record DirectForwardMetalHalfMatvecCoreOptions(
        Boolean enableMetalHalfMatvec,
        boolean disableMetalHalfMatvec,
        Boolean autoMetalHalfMatvec,
        int metalHalfMatvecMaxOutput) {

    private static final String ENABLE_METAL_HALF_MATVEC_PROPERTY =
            "gollek.safetensor.enable_metal_half_matvec";
    private static final String DISABLE_METAL_HALF_MATVEC_PROPERTY =
            "gollek.safetensor.disable_metal_half_matvec";
    private static final String AUTO_METAL_HALF_MATVEC_PROPERTY =
            "gollek.safetensor.auto_metal_half_matvec";
    private static final String METAL_HALF_MATVEC_MAX_OUTPUT_PROPERTY =
            "gollek.safetensor.metal_half_matvec_max_output";
    private static final int DEFAULT_METAL_HALF_MATVEC_MAX_OUTPUT = 24576;

    static DirectForwardMetalHalfMatvecCoreOptions fromSystemProperties() {
        return new DirectForwardMetalHalfMatvecCoreOptions(
                parseOptionalBoolean(System.getProperty(ENABLE_METAL_HALF_MATVEC_PROPERTY)),
                Boolean.getBoolean(DISABLE_METAL_HALF_MATVEC_PROPERTY),
                parseOptionalBoolean(System.getProperty(AUTO_METAL_HALF_MATVEC_PROPERTY)),
                Integer.getInteger(METAL_HALF_MATVEC_MAX_OUTPUT_PROPERTY, DEFAULT_METAL_HALF_MATVEC_MAX_OUTPUT));
    }

    static DirectForwardMetalHalfMatvecCoreOptions defaults() {
        return new DirectForwardMetalHalfMatvecCoreOptions(
                null,
                false,
                null,
                DEFAULT_METAL_HALF_MATVEC_MAX_OUTPUT);
    }

    DirectForwardMetalHalfMatvecCoreOptions withHalfMatvec(Boolean enable, boolean disable, Boolean auto,
            int maxOutput) {
        return new DirectForwardMetalHalfMatvecCoreOptions(enable, disable, auto, maxOutput);
    }
}
