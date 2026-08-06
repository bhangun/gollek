/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.planning;

import java.util.Objects;

record InferencePromptPlan(
        PreparedPrompt preparedPrompt,
        String ttsPrompt) {

    InferencePromptPlan {
        Objects.requireNonNull(preparedPrompt, "preparedPrompt");
        Objects.requireNonNull(ttsPrompt, "ttsPrompt");
    }
}
