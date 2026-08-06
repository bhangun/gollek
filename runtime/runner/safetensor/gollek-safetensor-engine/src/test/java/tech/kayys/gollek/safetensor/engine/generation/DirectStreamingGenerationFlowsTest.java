/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation;

import org.junit.jupiter.api.Test;
import tech.kayys.gollek.safetensor.generation.GenerationConfig;
import tech.kayys.gollek.spi.inference.InferenceResponse;

import java.nio.file.Path;
import java.time.Instant;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;

class DirectStreamingGenerationFlowsTest {

    @Test
    void continuationRequiresSessionBeforeResolvingModelOrWorkers() {
        AtomicBoolean resolvedModel = new AtomicBoolean(false);
        AtomicBoolean resolvedSessionAllocator = new AtomicBoolean(false);
        AtomicBoolean resolvedExecutor = new AtomicBoolean(false);
        DirectStreamingGenerationFlows flows = new DirectStreamingGenerationFlows((path, verbose, debugPrefix) -> {
            resolvedModel.set(true);
            throw new AssertionError("model should not be resolved without a session");
        }, () -> {
            resolvedSessionAllocator.set(true);
            throw new AssertionError("session allocator should not be resolved without a session");
        }, () -> {
            resolvedExecutor.set(true);
            throw new AssertionError("executor should not be resolved without a session");
        });

        IllegalArgumentException thrown = assertThrows(IllegalArgumentException.class,
                () -> flows.continuationTrace(
                        requestContext(), new long[] { 1, 2 }, 1, null, Path.of("model"),
                        GenerationConfig.defaults(), null, null, sink()));

        assertEquals("Conversation continuation requires an active KV cache session", thrown.getMessage());
        assertFalse(resolvedModel.get());
        assertFalse(resolvedSessionAllocator.get());
        assertFalse(resolvedExecutor.get());
    }

    @Test
    void textStreamRequiresSinkBeforeResolvingModel() {
        AtomicBoolean resolvedModel = new AtomicBoolean(false);
        DirectStreamingGenerationFlows flows = new DirectStreamingGenerationFlows((path, verbose, debugPrefix) -> {
            resolvedModel.set(true);
            return null;
        }, () -> {
            throw new AssertionError("session allocator should not be resolved without a sink");
        }, () -> {
            throw new AssertionError("executor should not be resolved without a sink");
        });

        assertThrows(NullPointerException.class,
                () -> flows.textResponse(
                        requestContext(), "prompt", Path.of("model"), GenerationConfig.defaults(), null));

        assertFalse(resolvedModel.get());
    }

    private static DirectStreamingGenerationFlows.StreamSink sink() {
        return new DirectStreamingGenerationFlows.StreamSink() {
            @Override
            public void emit(InferenceResponse response) {
            }

            @Override
            public boolean isCancelled() {
                return false;
            }
        };
    }

    private static DirectGenerationRequestContext requestContext() {
        return new DirectGenerationRequestContext(
                "request",
                Instant.EPOCH,
                1L,
                new InferenceProfile("test", false),
                "cpu",
                new DirectGenerationTimings());
    }
}
