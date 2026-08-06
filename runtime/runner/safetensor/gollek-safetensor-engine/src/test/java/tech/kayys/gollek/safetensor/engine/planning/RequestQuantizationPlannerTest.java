/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.planning;

import org.junit.jupiter.api.Test;
import tech.kayys.alkhawarizm.safetensor.quantization.QuantizationEngine;
import tech.kayys.gollek.spi.Message;
import tech.kayys.gollek.spi.provider.ProviderRequest;

import static org.junit.jupiter.api.Assertions.assertEquals;

class RequestQuantizationPlannerTest {
    private final RequestQuantizationPlanner planner = new RequestQuantizationPlanner();

    @Test
    void missingQuantizationStrategyDefaultsToNone() {
        assertEquals(
                QuantizationEngine.QuantStrategy.NONE,
                planner.resolve(request().build()));
    }

    @Test
    void blankQuantizationStrategyDefaultsToNone() {
        assertEquals(
                QuantizationEngine.QuantStrategy.NONE,
                planner.resolve(request()
                        .parameter("quantize_strategy", " ")
                        .build()));
    }

    @Test
    void strategyParsingIgnoresCaseAndSurroundingWhitespace() {
        assertEquals(
                QuantizationEngine.QuantStrategy.INT4,
                planner.resolve(request()
                        .parameter("quantize_strategy", " int4 ")
                        .build()));
    }

    @Test
    void unknownQuantizationStrategyFallsBackToNone() {
        assertEquals(
                QuantizationEngine.QuantStrategy.NONE,
                planner.resolve(request()
                        .parameter("quantize_strategy", "experimental")
                        .build()));
    }

    private static ProviderRequest.Builder request() {
        return ProviderRequest.builder()
                .model("unit-test-model")
                .message(Message.user("hello"));
    }
}
