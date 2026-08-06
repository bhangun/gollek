/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.weights;

import org.junit.jupiter.api.Test;
import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;

class WeightTensorResolverTest {

    @Test
    void firstCandidateWinsBeforeFallbacks() {
        AccelTensor primary = AccelTensor.zeros(1);
        AccelTensor fallback = AccelTensor.zeros(1);
        Map<String, AccelTensor> weights = Map.of(
                "candidate.primary", primary,
                "fallback.primary", fallback);

        assertSame(primary, WeightTensorResolver.first(
                weights,
                List.of("candidate.primary"),
                "fallback.primary"));
    }

    @Test
    void fallbackCandidatesAreUsedWhenDeclaredCandidatesAreMissing() {
        AccelTensor fallback = AccelTensor.zeros(1);
        Map<String, AccelTensor> weights = Map.of("fallback.primary", fallback);

        assertSame(fallback, WeightTensorResolver.first(
                weights,
                List.of("candidate.missing"),
                "fallback.primary"));
    }

    @Test
    void nullInputsResolveToNull() {
        assertNull(WeightTensorResolver.first(null, List.of("candidate")));
        assertNull(WeightTensorResolver.first(Map.of(), null, "fallback"));
        assertNull(WeightTensorResolver.tensor(Map.of(), null));
    }
}
