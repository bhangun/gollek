/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

/**
 * Accumulates generation timings and mirrors the same counters into the active
 * profile where the lower-level loop does not already do it.
 */
final class DirectGenerationTimings {
    private long sessionAllocateNanos;
    private long prefillNanos;
    private long decodeNanos;
    private long samplingNanos;
    private long firstTokenNanos;
    private int decodeSteps;

    void recordSessionAllocate(long nanos, InferenceProfile profile) {
        long elapsed = Math.max(0L, nanos);
        sessionAllocateNanos += elapsed;
        if (profile != null) {
            profile.sessionAllocateNanos += elapsed;
        }
    }

    void recordPrefill(DirectGenerationStepSampler.StepResult prefill, InferenceProfile profile) {
        if (prefill == null) {
            return;
        }
        long forwardNanos = Math.max(0L, prefill.forwardNanos());
        long sampleNanos = Math.max(0L, prefill.samplingNanos());
        prefillNanos += forwardNanos;
        samplingNanos += sampleNanos;
        if (profile != null) {
            profile.prefillNanos += forwardNanos;
            profile.samplingNanos += sampleNanos;
        }
    }

    void recordLoop(DirectGenerationLoop.Result loop) {
        if (loop == null) {
            return;
        }
        firstTokenNanos = loop.firstTokenNanos();
        decodeNanos += loop.decodeNanos();
        decodeSteps += loop.decodeSteps();
        samplingNanos += loop.samplingNanos();
    }

    DirectInferenceResponses.BenchTimings benchTimings() {
        return new DirectInferenceResponses.BenchTimings(
                sessionAllocateNanos,
                prefillNanos,
                decodeNanos,
                samplingNanos,
                firstTokenNanos,
                decodeSteps);
    }
}
