/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import jakarta.enterprise.inject.Instance;

import java.time.Instant;
import java.util.UUID;

/**
 * Per-request generation state shared by sync and streaming direct inference
 * paths.
 */
record DirectGenerationRequestContext(
        String requestId,
        Instant startedAt,
        long startedNanos,
        InferenceProfile profile,
        String backend,
        DirectGenerationTimings timings) {

    static DirectGenerationRequestContext sync(Instance<?> metalBackend) {
        return create("sync", metalBackend);
    }

    static DirectGenerationRequestContext stream(Instance<?> metalBackend) {
        return create("stream", metalBackend);
    }

    DirectInferenceResponses.BenchTimings benchTimings() {
        return timings.benchTimings();
    }

    private static DirectGenerationRequestContext create(String profileMode, Instance<?> metalBackend) {
        return new DirectGenerationRequestContext(
                UUID.randomUUID().toString(),
                Instant.now(),
                System.nanoTime(),
                DirectInferenceProfiler.startProfile(profileMode),
                DirectInferenceProfiler.backendLabel(metalBackend),
                new DirectGenerationTimings());
    }
}
