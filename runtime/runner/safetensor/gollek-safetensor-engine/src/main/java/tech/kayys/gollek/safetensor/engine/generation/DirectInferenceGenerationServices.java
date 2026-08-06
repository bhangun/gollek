/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import tech.kayys.gollek.safetensor.engine.forward.DirectForwardPass;
import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;

import java.util.Objects;
import java.util.function.Supplier;

/**
 * Owns the lazy dependency graph for direct generation execution.
 */
final class DirectInferenceGenerationServices {
    private final Supplier<DirectForwardPass> forwardPass;
    private final Supplier<TokenSampler> tokenSampler;
    private final Supplier<KVCacheManager> kvCacheManager;
    private final DirectGenerationFlows.LoadedModelResolver modelResolver;

    private DirectGenerationStepSampler stepSampler;
    private DirectGenerationLoop generationLoop;
    private DirectGenerationExecutor generationExecutor;
    private DirectGenerationSessionAllocator sessionAllocator;
    private DirectGenerationFlows generationFlows;
    private DirectStreamingGenerationFlows streamingGenerationFlows;

    DirectInferenceGenerationServices(Supplier<DirectForwardPass> forwardPass,
            Supplier<TokenSampler> tokenSampler,
            Supplier<KVCacheManager> kvCacheManager,
            DirectGenerationFlows.LoadedModelResolver modelResolver) {
        this.forwardPass = Objects.requireNonNull(forwardPass, "forwardPass");
        this.tokenSampler = Objects.requireNonNull(tokenSampler, "tokenSampler");
        this.kvCacheManager = Objects.requireNonNull(kvCacheManager, "kvCacheManager");
        this.modelResolver = Objects.requireNonNull(modelResolver, "modelResolver");
    }

    DirectGenerationFlows generationFlows() {
        if (generationFlows == null) {
            generationFlows = new DirectGenerationFlows(
                    modelResolver,
                    this::sessionAllocator,
                    this::generationExecutor);
        }
        return generationFlows;
    }

    DirectStreamingGenerationFlows streamingGenerationFlows() {
        if (streamingGenerationFlows == null) {
            streamingGenerationFlows = new DirectStreamingGenerationFlows(
                    modelResolver,
                    this::sessionAllocator,
                    this::generationExecutor);
        }
        return streamingGenerationFlows;
    }

    private DirectGenerationStepSampler stepSampler() {
        if (stepSampler == null) {
            stepSampler = new DirectGenerationStepSampler(forwardPass, tokenSampler);
        }
        return stepSampler;
    }

    private DirectGenerationLoop generationLoop() {
        if (generationLoop == null) {
            generationLoop = new DirectGenerationLoop(this::stepSampler);
        }
        return generationLoop;
    }

    private DirectGenerationExecutor generationExecutor() {
        if (generationExecutor == null) {
            generationExecutor = new DirectGenerationExecutor(this::stepSampler, this::generationLoop);
        }
        return generationExecutor;
    }

    private DirectGenerationSessionAllocator sessionAllocator() {
        if (sessionAllocator == null) {
            sessionAllocator = new DirectGenerationSessionAllocator(kvCacheManager);
        }
        return sessionAllocator;
    }
}
