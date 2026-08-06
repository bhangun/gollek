/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.generation.attention;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;

import java.lang.foreign.MemorySegment;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FlashAttentionPlanTest {
    @Test
    void resolvesPackedQkvPlanAndDispatchRequestFields() throws Exception {
        ModelConfig config = new ObjectMapper().readValue("""
                {
                  "model_type": "phi3",
                  "architectures": ["Phi3ForCausalLM"],
                  "hidden_size": 3072,
                  "num_attention_heads": 24,
                  "num_key_value_heads": 8
                }
                """, ModelConfig.class);
        AccelTensor x = AccelTensor.view(MemorySegment.NULL, new long[] { 1, 9, 3072 });
        AccelTensor sharedPackedQkvWeight = AccelTensor.view(MemorySegment.NULL, new long[] { 5120, 3072 });
        AttentionInput input = new AttentionInput(
                x, sharedPackedQkvWeight, sharedPackedQkvWeight, sharedPackedQkvWeight, null,
                null, null, null, null,
                null, config, null, 0, 7, true,
                null, null, null, null);

        FlashAttentionPlan plan = FlashAttentionPlan.resolve(input, new FlashAttentionKvCacheStage());

        assertSame(config, plan.config());
        assertEquals(0, plan.layerIdx());
        assertEquals(7, plan.startPos());
        assertEquals(9, plan.seqLen());
        assertEquals(24, plan.numQueryHeads());
        assertEquals(8, plan.numKeyValueHeads());
        assertEquals(128, plan.headDim());
        assertTrue(plan.headLayout().packedQkvProjection());
        assertFalse(plan.sharedKv());
        assertFalse(plan.useDenseSharedKvState());

        AccelTensor q = AccelTensor.view(MemorySegment.NULL, new long[] { 1, 9, 24, 128 });
        AccelTensor k = AccelTensor.view(MemorySegment.NULL, new long[] { 1, 9, 8, 128 });
        AccelTensor v = AccelTensor.view(MemorySegment.NULL, new long[] { 1, 9, 8, 128 });
        FlashAttentionDispatchRequest request = plan.dispatchRequest(q, k, v, null, true, null);

        assertSame(q, request.query());
        assertSame(k, request.key());
        assertSame(v, request.value());
        assertEquals(plan.layerIdx(), request.layerIdx());
        assertEquals(plan.kvLayerIdx(), request.kvLayerIdx());
        assertEquals(plan.startPos(), request.startPos());
        assertEquals(plan.seqLen(), request.seqLen());
        assertEquals(plan.numQueryHeads(), request.numQueryHeads());
        assertEquals(plan.numKeyValueHeads(), request.numKeyValueHeads());
        assertEquals(plan.headDim(), request.headDim());
        assertEquals(plan.attentionScale(), request.scale());
        assertEquals(plan.attentionSoftCap(), request.attentionSoftCap());
        assertTrue(request.causal());
        assertFalse(request.useDenseSharedKvState());
    }

    @Test
    void injectedNormalizationOptionsFlowIntoResolvedPlan() throws Exception {
        ModelConfig config = new ObjectMapper().readValue("""
                {
                  "model_type": "gemma4_text",
                  "architectures": ["Gemma4ForCausalLM"],
                  "hidden_size": 256,
                  "num_attention_heads": 4,
                  "num_key_value_heads": 2
                }
                """, ModelConfig.class);
        AccelTensor x = AccelTensor.view(MemorySegment.NULL, new long[] { 1, 1, 256 });
        AccelTensor weight = AccelTensor.view(MemorySegment.NULL, new long[] { 256, 256 });
        AttentionInput input = new AttentionInput(
                x, weight, weight, weight, null,
                null, null, null, null,
                null, config, null, 0, 0, true,
                null, null, null, null);

        FlashAttentionPlan plan = FlashAttentionPlan.resolve(
                input, new FlashAttentionKvCacheStage(), new FlashAttentionNormalizationOptions(true, true));

        assertFalse(plan.normalizationPolicy().qkNormEnabled());
        assertFalse(plan.normalizationPolicy().valueNormEnabled());
    }
}
