/*
 * Gollek Inference Engine - SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 */
package tech.kayys.gollek.safetensor.engine.forward;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import tech.kayys.gollek.models.gemma4.NativeBf16Family;
import tech.kayys.alkhawarizm.safetensor.core.tensor.AccelTensor;
import tech.kayys.alkhawarizm.spi.model.ModelArchitecture;
import tech.kayys.alkhawarizm.spi.model.ModelConfig;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;

class ResolvedModelWeightsCandidateTest {

    @Test
    void resolvesModelAndLayerWeightsFromArchitectureCandidates() throws Exception {
        ModelConfig config = new ObjectMapper().readValue("""
                {
                  "model_type": "candidate-test",
                  "num_hidden_layers": 1,
                  "tie_word_embeddings": false
                }
                """, ModelConfig.class);
        AccelTensor embed = AccelTensor.zeros(4, 3);
        AccelTensor lmHead = AccelTensor.zeros(4, 3);
        AccelTensor finalNorm = AccelTensor.zeros(3);
        AccelTensor query = AccelTensor.zeros(3, 3);
        AccelTensor attentionNorm = AccelTensor.zeros(3);
        AccelTensor ffnNorm = AccelTensor.zeros(3);
        AccelTensor layerScalar = AccelTensor.zeros(1);
        layerScalar.setFlat(0, 0.75f);

        Map<String, AccelTensor> weights = new HashMap<>();
        weights.put("fallback.embed", embed);
        weights.put("fallback.lm_head", lmHead);
        weights.put("fallback.final_norm", finalNorm);
        weights.put("fallback.layer.0.q", query);
        weights.put("fallback.layer.0.attn_norm", attentionNorm);
        weights.put("fallback.layer.0.ffn_norm", ffnNorm);
        weights.put("fallback.layer.0.scalar", layerScalar);

        ResolvedModelWeights resolved = ResolvedModelWeights.create(
                weights, config, new CandidateArchitecture(), false);

        assertSame(embed, resolved.embedTokens());
        assertSame(lmHead, resolved.lmHead());
        assertSame(finalNorm, resolved.finalNorm());
        assertSame(query, resolved.layer(0).queryWeight());
        assertSame(attentionNorm, resolved.layer(0).attentionNormWeight());
        assertSame(ffnNorm, resolved.layer(0).preFfnNormWeight());
        assertSame(layerScalar, resolved.layer(0).layerScalarWeight());
        assertEquals(0.75f, resolved.layer(0).layerScalarValue(), 0.0001f);
    }

    @Test
    void resolvesFusedQueryKeyValueWeightsFromFusedCandidates() throws Exception {
        ModelConfig config = new ObjectMapper().readValue("""
                {
                  "model_type": "fused-candidate-test",
                  "num_hidden_layers": 1
                }
                """, ModelConfig.class);
        AccelTensor fusedQkv = AccelTensor.zeros(8, 4);
        Map<String, AccelTensor> weights = Map.of("fallback.layer.0.qkv", fusedQkv);

        ResolvedModelWeights resolved = ResolvedModelWeights.create(
                weights, config, new FusedCandidateArchitecture(), false);

        assertSame(fusedQkv, resolved.layer(0).queryWeight());
        assertSame(fusedQkv, resolved.layer(0).keyWeight());
        assertSame(fusedQkv, resolved.layer(0).valueWeight());
    }

    @Test
    void resolvesNativeBf16UnifiedTextWrappedWeightsWithRealAdapter() throws Exception {
        ModelConfig config = new ObjectMapper().readValue("""
                {
                  "model_type": "gemma4_unified",
                  "architectures": ["NativeBf16UnifiedForConditionalGeneration"],
                  "hidden_size": 8,
                  "num_hidden_layers": 2,
                  "num_attention_heads": 2,
                  "num_key_value_heads": 1,
                  "num_global_key_value_heads": 1,
                  "intermediate_size": 16,
                  "vocab_size": 32,
                  "head_dim": 4,
                  "global_head_dim": 8,
                  "attention_k_eq_v": true,
                  "tie_word_embeddings": true,
                  "layer_types": ["sliding_attention", "full_attention"]
                }
                """, ModelConfig.class);
        Map<String, AccelTensor> weights = new HashMap<>();
        AccelTensor embed = put(weights, "model.language_model.embed_tokens.weight", 32, 8);
        AccelTensor finalNorm = put(weights, "model.language_model.norm.weight", 8);
        NativeBf16LayerTensors slidingLayer = putNativeBf16Layer(weights, 0, true, 8, 8, 4, 16);
        NativeBf16LayerTensors fullLayer = putNativeBf16Layer(weights, 1, false, 8, 16, 8, 16);

        ResolvedModelWeights resolved = ResolvedModelWeights.create(
                weights, config, new NativeBf16Family(), true);

        assertSame(embed, resolved.embedTokens());
        assertSame(embed, resolved.lmHead());
        assertSame(finalNorm, resolved.finalNorm());
        assertSame(slidingLayer.query(), resolved.layer(0).queryWeight());
        assertSame(slidingLayer.key(), resolved.layer(0).keyWeight());
        assertSame(slidingLayer.value(), resolved.layer(0).valueWeight());
        assertSame(slidingLayer.output(), resolved.layer(0).outputWeight());
        assertSame(slidingLayer.queryNorm(), resolved.layer(0).queryNormWeight());
        assertSame(slidingLayer.keyNorm(), resolved.layer(0).keyNormWeight());
        assertSame(slidingLayer.inputNorm(), resolved.layer(0).attentionNormWeight());
        assertSame(slidingLayer.postAttentionNorm(), resolved.layer(0).postAttnNormWeight());
        assertSame(slidingLayer.preFeedforwardNorm(), resolved.layer(0).preFfnNormWeight());
        assertSame(slidingLayer.postFeedforwardNorm(), resolved.layer(0).postFfnNormWeight());
        assertSame(slidingLayer.ffnGate(), resolved.layer(0).ffnGateWeight());
        assertSame(slidingLayer.ffnUp(), resolved.layer(0).ffnUpWeight());
        assertSame(slidingLayer.ffnDown(), resolved.layer(0).ffnDownWeight());
        assertSame(slidingLayer.layerScalar(), resolved.layer(0).layerScalarWeight());
        assertSame(fullLayer.query(), resolved.layer(1).queryWeight());
        assertSame(fullLayer.key(), resolved.layer(1).keyWeight());
        assertNull(resolved.layer(1).valueWeight());
        assertSame(fullLayer.output(), resolved.layer(1).outputWeight());
        assertSame(fullLayer.queryNorm(), resolved.layer(1).queryNormWeight());
        assertSame(fullLayer.keyNorm(), resolved.layer(1).keyNormWeight());
        assertSame(fullLayer.inputNorm(), resolved.layer(1).attentionNormWeight());
        assertSame(fullLayer.postAttentionNorm(), resolved.layer(1).postAttnNormWeight());
        assertSame(fullLayer.preFeedforwardNorm(), resolved.layer(1).preFfnNormWeight());
        assertSame(fullLayer.postFeedforwardNorm(), resolved.layer(1).postFfnNormWeight());
        assertSame(fullLayer.ffnGate(), resolved.layer(1).ffnGateWeight());
        assertSame(fullLayer.ffnUp(), resolved.layer(1).ffnUpWeight());
        assertSame(fullLayer.ffnDown(), resolved.layer(1).ffnDownWeight());
        assertSame(fullLayer.layerScalar(), resolved.layer(1).layerScalarWeight());
    }

    private static AccelTensor put(Map<String, AccelTensor> weights, String name, long... shape) {
        AccelTensor tensor = AccelTensor.zeros(shape);
        weights.put(name, tensor);
        return tensor;
    }

    private static NativeBf16LayerTensors putNativeBf16Layer(
            Map<String, AccelTensor> weights,
            int layer,
            boolean includeValueProjection,
            int hiddenSize,
            int queryRows,
            int keyValueRows,
            int intermediateSize) {
        String prefix = "model.language_model.layers.%d.".formatted(layer);
        AccelTensor layerScalar = put(weights, prefix + "layer_scalar", 1);
        AccelTensor inputNorm = put(weights, prefix + "input_layernorm.weight", hiddenSize);
        AccelTensor postAttentionNorm = put(weights, prefix + "post_attention_layernorm.weight", hiddenSize);
        AccelTensor preFeedforwardNorm = put(weights, prefix + "pre_feedforward_layernorm.weight", hiddenSize);
        AccelTensor postFeedforwardNorm = put(weights, prefix + "post_feedforward_layernorm.weight", hiddenSize);
        AccelTensor query = put(weights, prefix + "self_attn.q_proj.weight", queryRows, hiddenSize);
        AccelTensor key = put(weights, prefix + "self_attn.k_proj.weight", keyValueRows, hiddenSize);
        AccelTensor value = includeValueProjection
                ? put(weights, prefix + "self_attn.v_proj.weight", keyValueRows, hiddenSize)
                : null;
        AccelTensor output = put(weights, prefix + "self_attn.o_proj.weight", hiddenSize, queryRows);
        AccelTensor queryNorm = put(weights, prefix + "self_attn.q_norm.weight", queryRows / 2);
        AccelTensor keyNorm = put(weights, prefix + "self_attn.k_norm.weight", keyValueRows);
        AccelTensor ffnGate = put(weights, prefix + "mlp.gate_proj.weight", intermediateSize, hiddenSize);
        AccelTensor ffnUp = put(weights, prefix + "mlp.up_proj.weight", intermediateSize, hiddenSize);
        AccelTensor ffnDown = put(weights, prefix + "mlp.down_proj.weight", hiddenSize, intermediateSize);
        return new NativeBf16LayerTensors(
                query,
                key,
                value,
                output,
                inputNorm,
                postAttentionNorm,
                preFeedforwardNorm,
                postFeedforwardNorm,
                queryNorm,
                keyNorm,
                ffnGate,
                ffnUp,
                ffnDown,
                layerScalar);
    }

    /**
     * Bundles synthetic Gemma 4 layer tensors so the resolver test can assert every wrapped candidate explicitly.
     */
    private record NativeBf16LayerTensors(
            AccelTensor query,
            AccelTensor key,
            AccelTensor value,
            AccelTensor output,
            AccelTensor inputNorm,
            AccelTensor postAttentionNorm,
            AccelTensor preFeedforwardNorm,
            AccelTensor postFeedforwardNorm,
            AccelTensor queryNorm,
            AccelTensor keyNorm,
            AccelTensor ffnGate,
            AccelTensor ffnUp,
            AccelTensor ffnDown,
            AccelTensor layerScalar) {
    }

    private static final class CandidateArchitecture implements ModelArchitecture {
        @Override
        public String id() {
            return "candidate";
        }

        @Override
        public List<String> supportedArchClassNames() {
            return List.of("CandidateForCausalLM");
        }

        @Override
        public List<String> supportedModelTypes() {
            return List.of("candidate-test");
        }

        @Override
        public String embedTokensWeight() {
            return "primary.embed";
        }

        @Override
        public List<String> embedTokensWeightCandidates() {
            return List.of("primary.embed", "fallback.embed");
        }

        @Override
        public String finalNormWeight() {
            return "primary.final_norm";
        }

        @Override
        public List<String> finalNormWeightCandidates() {
            return List.of("primary.final_norm", "fallback.final_norm");
        }

        @Override
        public String lmHeadWeight() {
            return "primary.lm_head";
        }

        @Override
        public List<String> lmHeadWeightCandidates() {
            return List.of("primary.lm_head", "fallback.lm_head");
        }

        @Override
        public String layerQueryWeight(int i) {
            return "primary.layer.%d.q".formatted(i);
        }

        @Override
        public List<String> layerQueryWeightCandidates(int i) {
            return List.of(layerQueryWeight(i), "fallback.layer.%d.q".formatted(i));
        }

        @Override
        public String layerKeyWeight(int i) {
            return "primary.layer.%d.k".formatted(i);
        }

        @Override
        public String layerValueWeight(int i) {
            return "primary.layer.%d.v".formatted(i);
        }

        @Override
        public String layerOutputWeight(int i) {
            return "primary.layer.%d.o".formatted(i);
        }

        @Override
        public String layerAttentionNormWeight(int i) {
            return "primary.layer.%d.attn_norm".formatted(i);
        }

        @Override
        public List<String> layerAttentionNormWeightCandidates(int i) {
            return List.of(layerAttentionNormWeight(i), "fallback.layer.%d.attn_norm".formatted(i));
        }

        @Override
        public String layerFfnGateWeight(int i) {
            return "primary.layer.%d.ffn_gate".formatted(i);
        }

        @Override
        public String layerFfnUpWeight(int i) {
            return "primary.layer.%d.ffn_up".formatted(i);
        }

        @Override
        public String layerFfnDownWeight(int i) {
            return "primary.layer.%d.ffn_down".formatted(i);
        }

        @Override
        public String layerFfnNormWeight(int i) {
            return "primary.layer.%d.ffn_norm".formatted(i);
        }

        @Override
        public List<String> layerFfnNormWeightCandidates(int i) {
            return List.of(layerFfnNormWeight(i), "fallback.layer.%d.ffn_norm".formatted(i));
        }

        @Override
        public String layerScalarWeight(int i) {
            return "primary.layer.%d.scalar".formatted(i);
        }

        @Override
        public List<String> layerScalarWeightCandidates(int i) {
            return List.of(layerScalarWeight(i), "fallback.layer.%d.scalar".formatted(i));
        }
    }

    private static final class FusedCandidateArchitecture implements ModelArchitecture {
        @Override
        public String id() {
            return "fused-candidate";
        }

        @Override
        public List<String> supportedArchClassNames() {
            return List.of("FusedCandidateForCausalLM");
        }

        @Override
        public List<String> supportedModelTypes() {
            return List.of("fused-candidate-test");
        }

        @Override
        public String embedTokensWeight() {
            return "model.embed_tokens.weight";
        }

        @Override
        public String finalNormWeight() {
            return "model.norm.weight";
        }

        @Override
        public boolean hasFusedQKV() {
            return true;
        }

        @Override
        public String layerFusedQKVWeight(int i) {
            return "primary.layer.%d.qkv".formatted(i);
        }

        @Override
        public List<String> layerFusedQKVWeightCandidates(int i) {
            return List.of(layerFusedQKVWeight(i), "fallback.layer.%d.qkv".formatted(i));
        }

        @Override
        public String layerQueryWeight(int i) {
            return "unused.layer.%d.q".formatted(i);
        }

        @Override
        public String layerKeyWeight(int i) {
            return "unused.layer.%d.k".formatted(i);
        }

        @Override
        public String layerValueWeight(int i) {
            return "unused.layer.%d.v".formatted(i);
        }

        @Override
        public String layerOutputWeight(int i) {
            return "model.layers.%d.self_attn.o_proj.weight".formatted(i);
        }

        @Override
        public String layerAttentionNormWeight(int i) {
            return "model.layers.%d.input_layernorm.weight".formatted(i);
        }

        @Override
        public String layerFfnGateWeight(int i) {
            return "model.layers.%d.mlp.gate_proj.weight".formatted(i);
        }

        @Override
        public String layerFfnUpWeight(int i) {
            return "model.layers.%d.mlp.up_proj.weight".formatted(i);
        }

        @Override
        public String layerFfnDownWeight(int i) {
            return "model.layers.%d.mlp.down_proj.weight".formatted(i);
        }

        @Override
        public String layerFfnNormWeight(int i) {
            return "model.layers.%d.post_attention_layernorm.weight".formatted(i);
        }
    }
}
