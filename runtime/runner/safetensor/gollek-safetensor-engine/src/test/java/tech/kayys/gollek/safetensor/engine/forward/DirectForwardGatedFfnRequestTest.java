/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import org.jboss.logging.Logger;
import org.junit.jupiter.api.Test;
import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.kv.ForwardWorkspace;
import tech.kayys.alkhawarizm.spi.model.FFNActivationType;
import tech.kayys.alkhawarizm.spi.model.ModelArchitecture;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;

import java.lang.reflect.Proxy;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;

class DirectForwardGatedFfnRequestTest {

    @Test
    void delegatesExecutionStateAndWeightsFromFocusedBundles() {
        ModelConfig config = new ModelConfig();
        ModelConfigTraits traits = ModelConfigTraits.EMPTY;
        DirectForwardRuntimeContext runtime = runtime();
        ForwardWorkspace workspace = new ForwardWorkspace();
        AccelTensor input = AccelTensor.zeros(1, 1, 2);
        AccelTensor gateW = AccelTensor.zeros(2, 2);
        AccelTensor gateB = AccelTensor.zeros(2);
        AccelTensor upW = AccelTensor.zeros(2, 2);
        AccelTensor upB = AccelTensor.zeros(2);
        AccelTensor downW = AccelTensor.zeros(2, 2);
        AccelTensor downB = AccelTensor.zeros(2);
        AccelTensor downOutput = AccelTensor.zeros(1, 1, 2);

        try {
            DirectForwardGatedFfnRequest request = DirectForwardGatedFfnRequest.forward(
                    new DirectForwardLinearContext(runtime, traits, config),
                    architecture(FFNActivationType.GELU),
                    workspace,
                    input,
                    new DirectForwardGatedFfnWeights(gateW, gateB, upW, upB, downW, downB),
                    downOutput);

            assertSame(runtime, request.runtime());
            assertSame(traits, request.traits());
            assertSame(config, request.config());
            assertFalse(request.decodeLogitsPhase());
            assertSame(workspace, request.ws());
            assertSame(input, request.input());
            assertSame(gateW, request.gateW());
            assertSame(gateB, request.gateB());
            assertSame(upW, request.upW());
            assertSame(upB, request.upB());
            assertSame(downW, request.downW());
            assertSame(downB, request.downB());
            assertSame(downOutput, request.downOutputBuffer());
            assertEquals(FFNActivationType.GELU, request.activationType());
        } finally {
            input.close();
            gateW.close();
            gateB.close();
            upW.close();
            upB.close();
            downW.close();
            downB.close();
            downOutput.close();
            workspace.close();
        }
    }

    @Test
    void createsWeightBundleFromLayerWeightsWithoutReorderingTensors() {
        AccelTensor gateW = AccelTensor.zeros(2, 2);
        AccelTensor gateB = AccelTensor.zeros(2);
        AccelTensor upW = AccelTensor.zeros(2, 2);
        AccelTensor upB = AccelTensor.zeros(2);
        AccelTensor downW = AccelTensor.zeros(2, 2);
        AccelTensor downB = AccelTensor.zeros(2);

        try {
            ResolvedLayerWeights layerWeights = new ResolvedLayerWeights(
                    null, null, null, null,
                    null, null, null, null,
                    null,
                    gateW, gateB,
                    upW, upB,
                    downW, downB,
                    null, null, null, null, null,
                    null, null, null, null,
                    0.0f);

            DirectForwardGatedFfnWeights weights = DirectForwardGatedFfnWeights.fromLayer(layerWeights);

            assertSame(gateW, weights.gateW());
            assertSame(gateB, weights.gateB());
            assertSame(upW, weights.upW());
            assertSame(upB, weights.upB());
            assertSame(downW, weights.downW());
            assertSame(downB, weights.downB());
        } finally {
            gateW.close();
            gateB.close();
            upW.close();
            upB.close();
            downW.close();
            downB.close();
        }
    }

    private static DirectForwardRuntimeContext runtime() {
        return new DirectForwardRuntimeContext(
                Logger.getLogger(DirectForwardGatedFfnRequestTest.class),
                null,
                DirectForwardMetalCapabilities.EMPTY,
                false,
                false);
    }

    private static ModelArchitecture architecture(FFNActivationType activationType) {
        return (ModelArchitecture) Proxy.newProxyInstance(
                ModelArchitecture.class.getClassLoader(),
                new Class<?>[] { ModelArchitecture.class },
                (proxy, method, args) -> {
                    if ("activationType".equals(method.getName())) {
                        return activationType;
                    }
                    if (method.getReturnType() == String.class) {
                        return "test";
                    }
                    if (method.getReturnType() == List.class) {
                        return List.of();
                    }
                    if (method.getReturnType() == boolean.class) {
                        return false;
                    }
                    if (method.getReturnType() == int.class) {
                        return 0;
                    }
                    return null;
                });
    }
}
