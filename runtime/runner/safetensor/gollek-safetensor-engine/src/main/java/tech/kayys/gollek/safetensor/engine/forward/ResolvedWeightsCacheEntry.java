/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.alkhawarizm.spi.model.ModelArchitecture;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;

import java.util.Map;

record ResolvedWeightsCacheEntry(
        Map<String, AccelTensor> weights,
        ModelConfig config,
        ModelArchitecture arch,
        ResolvedModelWeights resolved) {
}
