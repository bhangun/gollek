package tech.kayys.gollek.safetensor.models;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelArchitecture;
import java.util.List;

/**
 * Qwen-2 architecture (0.5B, 1.5B, 7B, 14B, 32B, 72B, 110B).
 *
 * Key features:
 * - GQA (grouped query attention) with configurable num_key_value_heads
 * - SwiGLU FFN with intermediate_size = 4× hidden
 * - Dynamic NTK-aware RoPE scaling
 * - 151K vocabulary with multilingual BPE
 * - Q and K biases (unique: q_proj and k_proj have bias terms)
 *
 * HuggingFace models:
 * Qwen/Qwen2-0.5B-Instruct, Qwen/Qwen2-1.5B-Instruct
 * Qwen/Qwen2-7B-Instruct, Qwen/Qwen2-14B-Instruct
 * Qwen/Qwen2-32B-Instruct, Qwen/Qwen2-72B-Instruct
 */
@ApplicationScoped
public class Qwen2Family implements ModelArchitecture {

        @Override
        public String id() {
            return "qwen2";
        }

        @Override
        public List<String> supportedArchClassNames() {
            return List.of("Qwen2ForCausalLM");
        }

        @Override
        public List<String> supportedModelTypes() {
            return List.of("qwen2");
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
        public String lmHeadWeight() {
            return "lm_head.weight";
        }

        @Override
        public String layerQueryWeight(int i) {
            return "model.layers.%d.self_attn.q_proj.weight".formatted(i);
        }

        @Override
        public String layerKeyWeight(int i) {
            return "model.layers.%d.self_attn.k_proj.weight".formatted(i);
        }

        @Override
        public String layerValueWeight(int i) {
            return "model.layers.%d.self_attn.v_proj.weight".formatted(i);
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
        public String layerFfnNormWeight(int i) {
            return "model.layers.%d.post_attention_layernorm.weight".formatted(i);
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

        // Qwen-2 Q/K/V bias terms
        @Override
        public String layerQueryBias(int i) {
            return "model.layers.%d.self_attn.q_proj.bias".formatted(i);
        }

        @Override
        public String layerKeyBias(int i) {
            return "model.layers.%d.self_attn.k_proj.bias".formatted(i);
        }

        @Override
        public String layerValueBias(int i) {
            return "model.layers.%d.self_attn.v_proj.bias".formatted(i);
        }

        @Override
        public String layerOutputBias(int i) {
            return "model.layers.%d.self_attn.o_proj.bias".formatted(i);
        }

        @Override
        public String layerFfnGateBias(int i) {
            return "model.layers.%d.mlp.gate_proj.bias".formatted(i);
        }

        @Override
        public String layerFfnUpBias(int i) {
            return "model.layers.%d.mlp.up_proj.bias".formatted(i);
        }

        @Override
        public String layerFfnDownBias(int i) {
            return "model.layers.%d.mlp.down_proj.bias".formatted(i);
        }
}