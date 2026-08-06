/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import org.jboss.logging.Logger;
import org.junit.jupiter.api.Test;
import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DirectForwardLinearRequestTest {

    @Test
    void projectionFactoryCreatesForwardLinearRequest() {
        ModelConfig config = new ModelConfig();
        DirectForwardRuntimeContext runtime = runtime();
        DirectForwardLinearContext context =
                new DirectForwardLinearContext(runtime, ModelConfigTraits.EMPTY, config);
        AccelTensor input = AccelTensor.zeros(1, 1, 2);
        AccelTensor weight = AccelTensor.zeros(2, 2);
        AccelTensor bias = AccelTensor.zeros(2);
        AccelTensor outputBuffer = AccelTensor.zeros(1, 1, 2);

        try {
            DirectForwardLinearRequest request = DirectForwardLinearRequest.projection(
                    context,
                    input,
                    weight,
                    bias,
                    "ffn_up",
                    outputBuffer);

            assertSame(context, request.context());
            assertSame(runtime, request.runtime());
            assertSame(config, request.config());
            assertSame(ModelConfigTraits.EMPTY, request.traits());
            assertFalse(request.decodeLogitsPhase());
            assertSame(input, request.input());
            assertSame(weight, request.weight());
            assertSame(bias, request.bias());
            assertSame(outputBuffer, request.outputBuffer());
        } finally {
            input.close();
            weight.close();
            bias.close();
            outputBuffer.close();
        }
    }

    @Test
    void logitsFactoryPreservesDecodePhaseAndFallbackKeepsRuntime() {
        ModelConfig config = new ModelConfig();
        DirectForwardRuntimeContext runtime = runtime();
        DirectForwardLinearContext context =
                new DirectForwardLinearContext(runtime, ModelConfigTraits.EMPTY, config);
        AccelTensor input = AccelTensor.zeros(1, 1, 2);
        AccelTensor weight = AccelTensor.zeros(2, 2);
        AccelTensor replacementWeight = AccelTensor.zeros(2, 2);

        try {
            DirectForwardLinearRequest request = DirectForwardLinearRequest.logitsProjection(
                    context,
                    true,
                    input,
                    weight,
                    null,
                    "logits",
                    null);
            DirectForwardLinearRequest fallback = request.withFallbackContext(replacementWeight);

            assertTrue(request.decodeLogitsPhase());
            assertSame(context, request.context());
            assertSame(replacementWeight, fallback.weight());
            assertSame(runtime, fallback.runtime());
            assertSame(ModelConfigTraits.EMPTY, fallback.traits());
            assertNull(fallback.config());
            assertNull(fallback.outputBuffer());
            assertTrue(fallback.decodeLogitsPhase());
        } finally {
            input.close();
            weight.close();
            replacementWeight.close();
        }
    }

    @Test
    void gatedFfnFactoryReusesGatedLinearContextAndPhase() {
        ModelConfig config = new ModelConfig();
        DirectForwardRuntimeContext runtime = runtime();
        DirectForwardLinearContext linearContext =
                new DirectForwardLinearContext(runtime, ModelConfigTraits.EMPTY, config);
        DirectForwardGatedFfnRequest gatedRequest = new DirectForwardGatedFfnRequest(
                new DirectForwardGatedFfnContext(linearContext, true, null, null),
                null,
                new DirectForwardGatedFfnWeights(null, null, null, null, null, null),
                null);
        AccelTensor input = AccelTensor.zeros(1, 1, 2);
        AccelTensor weight = AccelTensor.zeros(2, 2);
        AccelTensor bias = AccelTensor.zeros(2);
        AccelTensor outputBuffer = AccelTensor.zeros(1, 1, 2);

        try {
            DirectForwardLinearRequest request = DirectForwardLinearRequest.gatedFfnProjection(
                    gatedRequest,
                    input,
                    weight,
                    bias,
                    "ffn_down",
                    outputBuffer);

            assertSame(linearContext, request.context());
            assertSame(runtime, request.runtime());
            assertSame(config, request.config());
            assertTrue(request.decodeLogitsPhase());
            assertSame(input, request.input());
            assertSame(weight, request.weight());
            assertSame(bias, request.bias());
            assertSame(outputBuffer, request.outputBuffer());
        } finally {
            input.close();
            weight.close();
            bias.close();
            outputBuffer.close();
        }
    }

    private static DirectForwardRuntimeContext runtime() {
        return new DirectForwardRuntimeContext(
                Logger.getLogger(DirectForwardLinearRequestTest.class),
                null,
                DirectForwardMetalCapabilities.EMPTY,
                false,
                false);
    }
}
