/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.attention.SharedKvState;
import tech.kayys.gollek.safetensor.engine.generation.kv.BlockManager;
import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DirectForwardSharedKvStatesTest {

    @Test
    void returnsNullWhenModelDoesNotUseSharedKvLayers() throws Exception {
        try (KVCacheManager.KVCacheSession session = session()) {
            assertNull(DirectForwardSharedKvStates.forPrefill(config(0), session, 0));
            assertNull(DirectForwardSharedKvStates.forDecode(config(0), session));
        }
    }

    @Test
    void clearsSharedKvStatesAtFreshPrefillOnly() throws Exception {
        ModelConfig config = config(2);
        try (KVCacheManager.KVCacheSession session = session()) {
            session.sharedKvStates().put(0, sharedState());

            Map<Integer, SharedKvState> freshPrefill =
                    DirectForwardSharedKvStates.forPrefill(config, session, 0);
            assertSame(session.sharedKvStates(), freshPrefill);
            assertTrue(freshPrefill.isEmpty());

            session.sharedKvStates().put(1, sharedState());
            Map<Integer, SharedKvState> continuationPrefill =
                    DirectForwardSharedKvStates.forPrefill(config, session, 3);
            assertSame(session.sharedKvStates(), continuationPrefill);
            assertSame(session.sharedKvStates().get(1), continuationPrefill.get(1));
        }
    }

    @Test
    void decodeReusesExistingSharedKvStates() throws Exception {
        ModelConfig config = config(1);
        try (KVCacheManager.KVCacheSession session = session()) {
            session.sharedKvStates().put(0, sharedState());

            Map<Integer, SharedKvState> decode = DirectForwardSharedKvStates.forDecode(config, session);

            assertSame(session.sharedKvStates(), decode);
            assertSame(session.sharedKvStates().get(0), decode.get(0));
        }
    }

    private static KVCacheManager.KVCacheSession session() {
        return new KVCacheManager.KVCacheSession(16, new BlockManager());
    }

    private static SharedKvState sharedState() {
        return new SharedKvState(AccelTensor.zeros(1, 1, 1, 1), AccelTensor.zeros(1, 1, 1, 1));
    }

    private static ModelConfig config(int sharedLayers) throws Exception {
        return new ObjectMapper().readValue("""
                {
                  "model_type": "shared-kv-test",
                  "hidden_size": 4,
                  "intermediate_size": 16,
                  "num_hidden_layers": 4,
                  "num_attention_heads": 2,
                  "num_key_value_heads": 1,
                  "num_kv_shared_layers": %d
                }
                """.formatted(sharedLayers), ModelConfig.class);
    }
}
