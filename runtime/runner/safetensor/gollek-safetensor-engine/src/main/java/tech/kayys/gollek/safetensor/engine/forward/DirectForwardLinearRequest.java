/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;

record DirectForwardLinearRequest(
        DirectForwardLinearContext context,
        boolean decodeLogitsPhase,
        AccelTensor input,
        AccelTensor weight,
        AccelTensor bias,
        String profileKey,
        AccelTensor outputBuffer) {

    static DirectForwardLinearRequest projection(DirectForwardLinearContext context,
                                                 AccelTensor input,
                                                 AccelTensor weight,
                                                 AccelTensor bias,
                                                 String profileKey,
                                                 AccelTensor outputBuffer) {
        return new DirectForwardLinearRequest(
                context,
                false,
                input,
                weight,
                bias,
                profileKey,
                outputBuffer);
    }

    static DirectForwardLinearRequest logitsProjection(DirectForwardLinearContext context,
                                                       boolean decodeLogitsPhase,
                                                       AccelTensor input,
                                                       AccelTensor weight,
                                                       AccelTensor bias,
                                                       String profileKey,
                                                       AccelTensor outputBuffer) {
        return new DirectForwardLinearRequest(
                context,
                decodeLogitsPhase,
                input,
                weight,
                bias,
                profileKey,
                outputBuffer);
    }

    static DirectForwardLinearRequest gatedFfnProjection(DirectForwardGatedFfnRequest request,
                                                         AccelTensor input,
                                                         AccelTensor weight,
                                                         AccelTensor bias,
                                                         String profileKey,
                                                         AccelTensor outputBuffer) {
        return new DirectForwardLinearRequest(
                request.context().linearContext(),
                request.decodeLogitsPhase(),
                input,
                weight,
                bias,
                profileKey,
                outputBuffer);
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

    DirectForwardLinearRequest withWeight(AccelTensor replacementWeight) {
        return new DirectForwardLinearRequest(
                context,
                decodeLogitsPhase,
                input,
                replacementWeight,
                bias,
                profileKey,
                outputBuffer);
    }

    DirectForwardLinearRequest withFallbackContext(AccelTensor replacementWeight) {
        return new DirectForwardLinearRequest(
                new DirectForwardLinearContext(runtime(), ModelConfigTraits.EMPTY, null),
                decodeLogitsPhase,
                input,
                replacementWeight,
                bias,
                profileKey,
                null);
    }
}
