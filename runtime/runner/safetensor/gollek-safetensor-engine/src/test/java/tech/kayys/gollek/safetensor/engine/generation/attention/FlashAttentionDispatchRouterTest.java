/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.safetensor.engine.generation.kv.BlockManager;
import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;

import static org.junit.jupiter.api.Assertions.assertEquals;

class FlashAttentionDispatchRouterTest {

    @Test
    void denseSharedRouteWinsBeforeOtherRoutingChecks() throws Exception {
        try (KVCacheManager.KVCacheSession session = session()) {
            assertEquals(FlashAttentionDispatchPath.DENSE_SHARED,
                    router(false).select(request(classicConfig(), session, 0, true)));
        }
    }

    @Test
    void classicAttentionUsesMetalTiledRouteWhenMetalIsAvailable() throws Exception {
        try (KVCacheManager.KVCacheSession session = session()) {
            assertEquals(FlashAttentionDispatchPath.METAL_TILED,
                    router(true).select(request(classicConfig(), session, 0, false)));
        }
    }

    @Test
    void classicAttentionFallsBackToPagedJavaWhenMetalIsUnavailable() throws Exception {
        try (KVCacheManager.KVCacheSession session = session()) {
            assertEquals(FlashAttentionDispatchPath.PAGED_JAVA,
                    router(false).select(request(classicConfig(), session, 0, false)));
        }
    }

    @Test
    void sharedKvLayerFallsBackToPagedJavaInsteadOfGeneralMetal() throws Exception {
        try (KVCacheManager.KVCacheSession session = session()) {
            assertEquals(FlashAttentionDispatchPath.PAGED_JAVA,
                    router(true).select(request(sharedKvConfig(), session, 1, false)));
        }
    }

    private static FlashAttentionDispatchRouter router(boolean canUseMetal) {
        FlashAttentionRoutingPolicy routing = new FlashAttentionRoutingPolicy(
                () -> canUseMetal,
                () -> null,
                () -> null,
                FlashAttentionRoutingOptions.defaults());
        return new FlashAttentionDispatchRouter(routing);
    }

    private static FlashAttentionDispatchRequest request(ModelConfig config,
            KVCacheManager.KVCacheSession session, int layerIdx, boolean useDenseSharedKvState) {
        return new FlashAttentionDispatchRequest(
                null,
                null,
                null,
                session,
                null,
                config,
                FlashAttentionModelPolicy.resolve(null, config),
                layerIdx,
                layerIdx,
                0,
                1,
                config.getNumAttentionHeads(),
                config.getResolvedNumKvHeadsForLayer(layerIdx),
                config.getResolvedHeadDim(),
                1.0f,
                true,
                0.0f,
                null,
                useDenseSharedKvState);
    }

    private static KVCacheManager.KVCacheSession session() {
        return new KVCacheManager.KVCacheSession(16, new BlockManager());
    }

    private static ModelConfig classicConfig() throws Exception {
        return new ObjectMapper().readValue("""
                {
                  "model_type": "phi3",
                  "hidden_size": 256,
                  "intermediate_size": 512,
                  "num_hidden_layers": 2,
                  "num_attention_heads": 4,
                  "num_key_value_heads": 2
                }
                """, ModelConfig.class);
    }

    private static ModelConfig sharedKvConfig() throws Exception {
        return new ObjectMapper().readValue("""
                {
                  "model_type": "phi3",
                  "hidden_size": 256,
                  "intermediate_size": 512,
                  "num_hidden_layers": 2,
                  "num_attention_heads": 4,
                  "num_key_value_heads": 2,
                  "num_kv_shared_layers": 1
                }
                """, ModelConfig.class);
    }
}
