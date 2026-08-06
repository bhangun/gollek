/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.gollek.safetensor.engine.generation.kv.BlockManager;
import tech.kayys.gollek.safetensor.engine.generation.kv.KVCacheManager;
import tech.kayys.gollek.safetensor.generation.GenerationConfig;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Verifies Gemma 4 projection-stage behavior that depends on alternative
 * attention layouts and reusable KV workspace buffers.
 */
class FlashAttentionProjectionStageTest {
    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

    @Test
    void gemma4AlternativeAttentionViewsValueFromKeyWhenVProjectionIsAbsent() throws Exception {
        ModelConfig config = gemma4AlternativeAttentionConfig();
        try (AccelTensor x = AccelTensor.fromFloatArray(new float[] {
                1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f
        }, 1, 1, 8);
                AccelTensor qW = AccelTensor.zeros(8, 8);
                AccelTensor kW = firstFourInputColumnsWeight()) {
            AttentionInput input = new AttentionInput(
                    x, qW, kW, null, null,
                    null, null, null, null,
                    null, config, null, 1, 0, true,
                    null, null, null, null);
            FlashAttentionPlan plan = FlashAttentionPlan.resolve(
                    input,
                    new FlashAttentionKvCacheStage(),
                    new FlashAttentionNormalizationOptions(true, true));

            assertTrue(plan.alternativeAttention());
            assertEquals(1, plan.numKeyValueHeads());
            assertEquals(4, plan.headDim());

            FlashAttentionProjectionStage stage = new FlashAttentionProjectionStage(
                    new FlashAttentionProjector(null, () -> false),
                    new FlashAttentionNormalizer(() -> null));
            FlashAttentionProjectionStage.PreparedTensors prepared = stage.prepare(
                    input,
                    config,
                    plan.modelPolicy(),
                    plan.headLayout(),
                    plan.normalizationPolicy(),
                    plan.sharedKvState(),
                    plan.sharedKv(),
                    plan.useDenseSharedKvState(),
                    plan.alternativeAttention());

            try {
                assertTrue(prepared.key().hasShape(new long[] { 1, 1, 1, 4 }));
                assertTrue(prepared.value().hasShape(new long[] { 1, 1, 1, 4 }));
                assertArrayEquals(new float[] { 1.0f, 2.0f, 3.0f, 4.0f },
                        prepared.key().toFloatArray(), 0.0001f);
                assertArrayEquals(prepared.key().toFloatArray(), prepared.value().toFloatArray(), 0.0001f);
                prepared.key().setFlat(0, 42.0f);
                assertEquals(42.0f, prepared.value().getFlat(0), 0.0001f);
            } finally {
                prepared.close();
            }
        }
    }

    @Test
    void gemma4AlternativeAttentionProjectsSingleTokenHalfQueryKeyIntoReusableWorkspaceWhenMetalIsUnavailable()
            throws Exception {
        assertHalfQueryKeyProjectedIntoWorkspace(1);
    }

    @Test
    void gemma4AlternativeAttentionProjectsSmallBatchHalfQueryKeyIntoReusableWorkspaceWhenMetalIsUnavailable()
            throws Exception {
        assertHalfQueryKeyProjectedIntoWorkspace(2);
    }

    @Test
    void gemma4AlternativeAttentionKeepsNormalizedQueryInReusableWorkspaceWhenMetalIsUnavailable() throws Exception {
        assertHalfQueryKeyProjectedIntoWorkspace(1, true, false);
    }

    @Test
    void gemma4AlternativeAttentionKeepsNormalizedValueInReusableWorkspaceAfterKeyIsSeparated() throws Exception {
        assertHalfQueryKeyProjectedIntoWorkspace(1, false, true);
    }

    @Test
    void preparedTensorsCloseReleasesQueryKeyValueAndSharedOwner() {
        FlashAttentionProjectionStage.PreparedTensors prepared =
                new FlashAttentionProjectionStage.PreparedTensors(
                        AccelTensor.zeros(1),
                        AccelTensor.zeros(1),
                        AccelTensor.zeros(1),
                        AccelTensor.zeros(1));

        assertFalse(prepared.query().isClosed());
        assertFalse(prepared.key().isClosed());
        assertFalse(prepared.value().isClosed());
        assertFalse(prepared.sharedOwner().isClosed());

        prepared.close();

        assertTrue(prepared.query().isClosed());
        assertTrue(prepared.key().isClosed());
        assertTrue(prepared.value().isClosed());
        assertTrue(prepared.sharedOwner().isClosed());
    }

    private static void assertHalfQueryKeyProjectedIntoWorkspace(int seqLen) throws Exception {
        assertHalfQueryKeyProjectedIntoWorkspace(seqLen, false, false);
    }

    private static void assertHalfQueryKeyProjectedIntoWorkspace(int seqLen, boolean queryNorm, boolean valueNorm)
            throws Exception {
        ModelConfig config = gemma4AlternativeAttentionConfig();
        try (KVCacheManager.KVCacheSession session = new KVCacheManager.KVCacheSession(16, new BlockManager());
                AccelTensor x = AccelTensor.fromFloatArray(sequence(seqLen * 8), 1, seqLen, 8);
                AccelTensor qW = halfZeros(8, 8);
                AccelTensor kW = halfZeros(4, 8);
                AccelTensor qNormW = queryNorm ? AccelTensor.ones(4) : null;
                AccelTensor kNormW = valueNorm ? AccelTensor.ones(4) : null) {
            session.allocate(config, GenerationConfig.defaults());
            AttentionInput input = new AttentionInput(
                    x, qW, kW, null, null,
                    null, null, null, null,
                    null, config, session, 1, 0, true,
                    qNormW, kNormW, null, null);
            FlashAttentionPlan plan = FlashAttentionPlan.resolve(
                    input,
                    new FlashAttentionKvCacheStage(),
                    new FlashAttentionNormalizationOptions(!(queryNorm || valueNorm), !valueNorm));

            FlashAttentionProjectionStage stage = new FlashAttentionProjectionStage(
                    new FlashAttentionProjector(null, () -> false),
                    new FlashAttentionNormalizer(() -> null));
            FlashAttentionProjectionStage.PreparedTensors prepared = stage.prepare(
                    input,
                    config,
                    plan.modelPolicy(),
                    plan.headLayout(),
                    plan.normalizationPolicy(),
                    plan.sharedKvState(),
                    plan.sharedKv(),
                    plan.useDenseSharedKvState(),
                    plan.alternativeAttention());

            try {
                MemorySegment scratch = session.getWorkspace().getCombinedSeg();
                long qBytes = prepared.query().numel() * (long) Float.BYTES;

                prepared.query().setFlat(0, 7.25f);
                assertEquals(7.25f, scratch.get(ValueLayout.JAVA_FLOAT, 0), 0.0001f);

                if (valueNorm) {
                    prepared.value().setFlat(0, 8.5f);
                    assertEquals(8.5f, scratch.get(ValueLayout.JAVA_FLOAT, qBytes), 0.0001f);
                } else {
                    prepared.key().setFlat(0, 8.5f);
                    assertEquals(8.5f, scratch.get(ValueLayout.JAVA_FLOAT, qBytes), 0.0001f);
                    prepared.key().setFlat(1, 9.75f);
                    assertEquals(9.75f, prepared.value().getFlat(1), 0.0001f);
                }
            } finally {
                prepared.close();
            }
        }
    }

    private static ModelConfig gemma4AlternativeAttentionConfig() throws Exception {
        return OBJECT_MAPPER.readValue("""
                {
                  "model_type": "gemma4_text",
                  "architectures": ["Gemma4ForCausalLM"],
                  "hidden_size": 8,
                  "intermediate_size": 32,
                  "num_hidden_layers": 2,
                  "num_attention_heads": 2,
                  "num_key_value_heads": 1,
                  "num_global_key_value_heads": 1,
                  "head_dim": 2,
                  "global_head_dim": 4,
                  "attention_k_eq_v": true,
                  "layer_types": ["sliding_attention", "full_attention"]
                }
                """, ModelConfig.class);
    }

    private static AccelTensor firstFourInputColumnsWeight() {
        float[] weights = new float[4 * 8];
        for (int row = 0; row < 4; row++) {
            weights[row * 8 + row] = 1.0f;
        }
        return AccelTensor.fromFloatArray(weights, 4, 8);
    }

    private static AccelTensor halfZeros(long... shape) {
        return AccelTensor.zeros(shape)
                .withQuantization(AccelTensor.QuantType.F16, null, null, -1);
    }

    private static float[] sequence(int length) {
        float[] values = new float[length];
        for (int i = 0; i < length; i++) {
            values[i] = i + 1.0f;
        }
        return values;
    }
}
