/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;
import tech.kayys.alkhawarizm.spi.model.ModelArchitecture;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;

import java.lang.foreign.MemorySegment;
import java.util.Map;

public class AttentionInput {
    public final AccelTensor x;
    public final AccelTensor qW, kW, vW, oW;
    public final AccelTensor qB, kB, vB, oB;
    public final ModelArchitecture arch;
    public final ModelConfig config;
    public final KVCacheManager.KVCacheSession kvCache;
    public final int layerIdx;
    public final int startPos;
    public final boolean isCausal;
    public final AccelTensor qNormW, kNormW;
    public final AccelTensor postAttnNormW;
    public final Map<Integer, SharedKvState> sharedKvStates;
    public final MemorySegment attentionContextBuffer;
    public final MemorySegment attentionOutputBuffer;

    public AttentionInput(AccelTensor x, AccelTensor qW, AccelTensor kW, AccelTensor vW, AccelTensor oW,
            AccelTensor qB, AccelTensor kB, AccelTensor vB, AccelTensor oB,
            ModelArchitecture arch, ModelConfig config, KVCacheManager.KVCacheSession kvCache,
            int layerIdx, int startPos, boolean isCausal,
            AccelTensor qNormW, AccelTensor kNormW, AccelTensor postAttnNormW,
            Map<Integer, SharedKvState> sharedKvStates) {
        this(x, qW, kW, vW, oW, qB, kB, vB, oB, arch, config, kvCache,
                layerIdx, startPos, isCausal, qNormW, kNormW, postAttnNormW, sharedKvStates, null);
    }

    public AttentionInput(AccelTensor x, AccelTensor qW, AccelTensor kW, AccelTensor vW, AccelTensor oW,
            AccelTensor qB, AccelTensor kB, AccelTensor vB, AccelTensor oB,
            ModelArchitecture arch, ModelConfig config, KVCacheManager.KVCacheSession kvCache,
            int layerIdx, int startPos, boolean isCausal,
            AccelTensor qNormW, AccelTensor kNormW, AccelTensor postAttnNormW,
            Map<Integer, SharedKvState> sharedKvStates, MemorySegment attentionOutputBuffer) {
        this(x, qW, kW, vW, oW, qB, kB, vB, oB, arch, config, kvCache,
                layerIdx, startPos, isCausal, qNormW, kNormW, postAttnNormW, sharedKvStates,
                null, attentionOutputBuffer);
    }

    public AttentionInput(AccelTensor x, AccelTensor qW, AccelTensor kW, AccelTensor vW, AccelTensor oW,
            AccelTensor qB, AccelTensor kB, AccelTensor vB, AccelTensor oB,
            ModelArchitecture arch, ModelConfig config, KVCacheManager.KVCacheSession kvCache,
            int layerIdx, int startPos, boolean isCausal,
            AccelTensor qNormW, AccelTensor kNormW, AccelTensor postAttnNormW,
            Map<Integer, SharedKvState> sharedKvStates, MemorySegment attentionContextBuffer,
            MemorySegment attentionOutputBuffer) {
        this.x = x;
        this.qW = qW;
        this.kW = kW;
        this.vW = vW;
        this.oW = oW;
        this.qB = qB;
        this.kB = kB;
        this.vB = vB;
        this.oB = oB;
        this.arch = arch;
        this.config = config;
        this.kvCache = kvCache;
        this.layerIdx = layerIdx;
        this.startPos = startPos;
        this.isCausal = isCausal;
        this.qNormW = qNormW;
        this.kNormW = kNormW;
        this.postAttnNormW = postAttnNormW;
        this.sharedKvStates = sharedKvStates;
        this.attentionContextBuffer = attentionContextBuffer;
        this.attentionOutputBuffer = attentionOutputBuffer;
    }
}
