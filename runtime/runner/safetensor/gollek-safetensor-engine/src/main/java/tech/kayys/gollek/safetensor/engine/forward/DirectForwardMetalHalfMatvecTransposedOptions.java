/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import static tech.kayys.gollek.safetensor.engine.forward.DirectForwardRuntimeOptions.parseOptionalBoolean;

record DirectForwardMetalHalfMatvecTransposedOptions(
        Boolean enableMetalTransposedHalfMatvec,
        boolean enableMetalTransposedHalfMatvecAll,
        boolean disableMetalTransposedHalfMatvec,
        int metalTransposedHalfMatvecMaxOutput) {

    private static final String ENABLE_METAL_TRANSPOSED_HALF_MATVEC_PROPERTY =
            "gollek.safetensor.enable_metal_transposed_half_matvec";
    private static final String ENABLE_METAL_TRANSPOSED_HALF_MATVEC_ALL_PROPERTY =
            "gollek.safetensor.enable_metal_transposed_half_matvec_all";
    private static final String DISABLE_METAL_TRANSPOSED_HALF_MATVEC_PROPERTY =
            "gollek.safetensor.disable_metal_transposed_half_matvec";
    private static final String METAL_TRANSPOSED_HALF_MATVEC_MAX_OUTPUT_PROPERTY =
            "gollek.safetensor.metal_transposed_half_matvec_max_output";
    private static final int DEFAULT_METAL_TRANSPOSED_HALF_MATVEC_MAX_OUTPUT = 300000;

    static DirectForwardMetalHalfMatvecTransposedOptions fromSystemProperties() {
        return new DirectForwardMetalHalfMatvecTransposedOptions(
                parseOptionalBoolean(System.getProperty(ENABLE_METAL_TRANSPOSED_HALF_MATVEC_PROPERTY)),
                Boolean.getBoolean(ENABLE_METAL_TRANSPOSED_HALF_MATVEC_ALL_PROPERTY),
                Boolean.getBoolean(DISABLE_METAL_TRANSPOSED_HALF_MATVEC_PROPERTY),
                Integer.getInteger(METAL_TRANSPOSED_HALF_MATVEC_MAX_OUTPUT_PROPERTY,
                        DEFAULT_METAL_TRANSPOSED_HALF_MATVEC_MAX_OUTPUT));
    }

    static DirectForwardMetalHalfMatvecTransposedOptions defaults() {
        return new DirectForwardMetalHalfMatvecTransposedOptions(
                null,
                false,
                false,
                DEFAULT_METAL_TRANSPOSED_HALF_MATVEC_MAX_OUTPUT);
    }

    DirectForwardMetalHalfMatvecTransposedOptions withTransposedHalfMatvec(
            Boolean enable,
            boolean allowAll,
            boolean disable,
            int maxOutput) {
        return new DirectForwardMetalHalfMatvecTransposedOptions(enable, allowAll, disable, maxOutput);
    }
}
