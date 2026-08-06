/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import java.util.Objects;

record DirectForwardMetalHalfMatvecOptions(
        DirectForwardMetalHalfMatvecPairOptions pairOptions,
        DirectForwardMetalHalfMatvecCoreOptions coreOptions,
        DirectForwardMetalHalfMatvecLogitsOptions logitsOptions,
        DirectForwardMetalHalfMatvecTransposedOptions transposedOptions) {

    DirectForwardMetalHalfMatvecOptions {
        pairOptions = Objects.requireNonNull(pairOptions, "pairOptions");
        coreOptions = Objects.requireNonNull(coreOptions, "coreOptions");
        logitsOptions = Objects.requireNonNull(logitsOptions, "logitsOptions");
        transposedOptions = Objects.requireNonNull(transposedOptions, "transposedOptions");
    }

    static DirectForwardMetalHalfMatvecOptions fromSystemProperties() {
        return new DirectForwardMetalHalfMatvecOptions(
                DirectForwardMetalHalfMatvecPairOptions.fromSystemProperties(),
                DirectForwardMetalHalfMatvecCoreOptions.fromSystemProperties(),
                DirectForwardMetalHalfMatvecLogitsOptions.fromSystemProperties(),
                DirectForwardMetalHalfMatvecTransposedOptions.fromSystemProperties());
    }

    static DirectForwardMetalHalfMatvecOptions defaults() {
        return new DirectForwardMetalHalfMatvecOptions(
                DirectForwardMetalHalfMatvecPairOptions.defaults(),
                DirectForwardMetalHalfMatvecCoreOptions.defaults(),
                DirectForwardMetalHalfMatvecLogitsOptions.defaults(),
                DirectForwardMetalHalfMatvecTransposedOptions.defaults());
    }

    DirectForwardMetalHalfMatvecOptions withHalfLinearPair(Boolean enable, boolean disable) {
        return new DirectForwardMetalHalfMatvecOptions(
                pairOptions.withHalfLinearPair(enable, disable),
                coreOptions,
                logitsOptions,
                transposedOptions);
    }

    DirectForwardMetalHalfMatvecOptions withHalfMatvec(Boolean enable, boolean disable, Boolean auto,
            int maxOutput) {
        return new DirectForwardMetalHalfMatvecOptions(
                pairOptions,
                coreOptions.withHalfMatvec(enable, disable, auto, maxOutput),
                logitsOptions,
                transposedOptions);
    }

    DirectForwardMetalHalfMatvecOptions withHalfMatvecPair(Boolean enable, boolean disable) {
        return new DirectForwardMetalHalfMatvecOptions(
                pairOptions.withHalfMatvecPair(enable, disable),
                coreOptions,
                logitsOptions,
                transposedOptions);
    }

    DirectForwardMetalHalfMatvecOptions withLogitsMpsMatvec(Boolean enable, boolean disable,
            int minOutput, int maxInput) {
        return new DirectForwardMetalHalfMatvecOptions(
                pairOptions,
                coreOptions,
                logitsOptions.withLogitsMpsMatvec(enable, disable, minOutput, maxInput),
                transposedOptions);
    }

    DirectForwardMetalHalfMatvecOptions withLogitsMaxOutputs(
            int gemma4MaxOutput, int gemma3MaxOutput, int qwenMaxOutput) {
        return new DirectForwardMetalHalfMatvecOptions(
                pairOptions,
                coreOptions,
                logitsOptions.withLogitsMaxOutputs(gemma4MaxOutput, gemma3MaxOutput, qwenMaxOutput),
                transposedOptions);
    }

    DirectForwardMetalHalfMatvecOptions withGemma3Logits(Boolean enable, boolean disable) {
        return new DirectForwardMetalHalfMatvecOptions(
                pairOptions,
                coreOptions,
                logitsOptions.withGemma3Logits(enable, disable),
                transposedOptions);
    }

    DirectForwardMetalHalfMatvecOptions withTransposedHalfMatvec(Boolean enable, boolean allowAll,
            boolean disable, int maxOutput) {
        return new DirectForwardMetalHalfMatvecOptions(
                pairOptions,
                coreOptions,
                logitsOptions,
                transposedOptions.withTransposedHalfMatvec(enable, allowAll, disable, maxOutput));
    }

}
