/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;
import tech.kayys.alkhawarizm.spi.model.ModelArchitecture;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;

import java.util.Map;

record DirectForwardDecodeRequest(
        long tokenId,
        int startPos,
        Map<String, AccelTensor> weights,
        ModelConfig config,
        ModelArchitecture arch,
        KVCacheManager.KVCacheSession kvCache,
        ResolvedModelWeights resolvedWeights,
        boolean reuseLogitsOutput) {
}
