/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import static tech.kayys.gollek.safetensor.engine.forward.DirectForwardRuntimeOptions.parseOptionalBoolean;

record DirectForwardMetalHalfMatvecPairOptions(
        Boolean enableMetalHalfLinearPair,
        boolean disableMetalHalfLinearPair,
        Boolean enableMetalHalfMatvecPair,
        boolean disableMetalHalfMatvecPair) {

    private static final String ENABLE_METAL_HALF_LINEAR_PAIR_PROPERTY =
            "gollek.safetensor.enable_metal_half_linear_pair";
    private static final String DISABLE_METAL_HALF_LINEAR_PAIR_PROPERTY =
            "gollek.safetensor.disable_metal_half_linear_pair";
    private static final String ENABLE_METAL_HALF_MATVEC_PAIR_PROPERTY =
            "gollek.safetensor.enable_metal_half_matvec_pair";
    private static final String DISABLE_METAL_HALF_MATVEC_PAIR_PROPERTY =
            "gollek.safetensor.disable_metal_half_matvec_pair";

    static DirectForwardMetalHalfMatvecPairOptions fromSystemProperties() {
        return new DirectForwardMetalHalfMatvecPairOptions(
                parseOptionalBoolean(System.getProperty(ENABLE_METAL_HALF_LINEAR_PAIR_PROPERTY)),
                Boolean.getBoolean(DISABLE_METAL_HALF_LINEAR_PAIR_PROPERTY),
                parseOptionalBoolean(System.getProperty(ENABLE_METAL_HALF_MATVEC_PAIR_PROPERTY)),
                Boolean.getBoolean(DISABLE_METAL_HALF_MATVEC_PAIR_PROPERTY));
    }

    static DirectForwardMetalHalfMatvecPairOptions defaults() {
        return new DirectForwardMetalHalfMatvecPairOptions(null, false, null, false);
    }

    DirectForwardMetalHalfMatvecPairOptions withHalfLinearPair(Boolean enable, boolean disable) {
        return new DirectForwardMetalHalfMatvecPairOptions(
                enable,
                disable,
                enableMetalHalfMatvecPair,
                disableMetalHalfMatvecPair);
    }

    DirectForwardMetalHalfMatvecPairOptions withHalfMatvecPair(Boolean enable, boolean disable) {
        return new DirectForwardMetalHalfMatvecPairOptions(
                enableMetalHalfLinearPair,
                disableMetalHalfLinearPair,
                enable,
                disable);
    }
}
