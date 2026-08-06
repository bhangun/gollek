/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.kv.ForwardWorkspace;
import tech.kayys.alkhawarizm.spi.model.FFNActivationType;
import tech.kayys.alkhawarizm.spi.model.ModelArchitecture;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;

record DirectForwardGatedFfnRequest(
        DirectForwardGatedFfnContext context,
        AccelTensor input,
        DirectForwardGatedFfnWeights weights,
        AccelTensor downOutputBuffer) {

    static DirectForwardGatedFfnRequest forward(
            DirectForwardLinearContext linearContext,
            ModelArchitecture arch,
            ForwardWorkspace workspace,
            AccelTensor input,
            DirectForwardGatedFfnWeights weights,
            AccelTensor downOutputBuffer) {
        return new DirectForwardGatedFfnRequest(
                new DirectForwardGatedFfnContext(linearContext, false, arch, workspace),
                input,
                weights,
                downOutputBuffer);
    }

    DirectForwardRuntimeContext runtime() {
        return context.runtime();
    }

    ModelConfigTraits traits() {
        return context.traits();
    }

    ModelConfig config() {
        return context.config();
    }

    boolean decodeLogitsPhase() {
        return context.decodeLogitsPhase();
    }

    AccelTensor gateW() {
        return weights.gateW();
    }

    AccelTensor gateB() {
        return weights.gateB();
    }

    AccelTensor upW() {
        return weights.upW();
    }

    AccelTensor upB() {
        return weights.upB();
    }

    AccelTensor downW() {
        return weights.downW();
    }

    AccelTensor downB() {
        return weights.downB();
    }

    ForwardWorkspace ws() {
        return context.workspace();
    }

    boolean metalLinearEnabled() {
        return context.metalLinearEnabled();
    }

    FFNActivationType activationType() {
        return context.activationType();
    }
}
