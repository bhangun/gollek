/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import tech.kayys.gollek.safetensor.engine.generation.kv.ForwardWorkspace;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;

record DirectForwardOutputProjectionContext(
        DirectForwardRuntimeContext runtime,
        ModelConfigTraits traits,
        ModelConfig config,
        ResolvedModelWeights resolvedWeights,
        ForwardWorkspace workspace) {
}
