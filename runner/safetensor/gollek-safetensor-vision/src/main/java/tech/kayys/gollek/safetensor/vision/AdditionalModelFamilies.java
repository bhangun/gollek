/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * AdditionalModelFamilies.java
 * ─────────────────────────────
 * Additional open-weight model architecture beans for 2025/2026 releases.
 *
 * Covered families:
 *   Cohere Command-R / Command-R+  — enterprise RAG model (Cohere)
 *   DeepSeek-V3                    — improved MoE with Multi-head Latent Attention
 *   Gemma-3 (text-only)            — Google's 2025 Gemma update
 *   OLMo-2                         — Allen AI open language model v2
 *   Mistral-NeMo                   — Mistral × NVIDIA 12B model
 *   SmolLM / SmolLM2               — HuggingFace compact models (135M–1.7B)
 *   Phi-4                          — Microsoft Phi-4-mini and Phi-4
 */
package tech.kayys.gollek.safetensor.vision;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.spi.model.ModelArchitecture;

import java.util.List;

public final class AdditionalModelFamilies {

    private AdditionalModelFamilies() {
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Cohere Command-R / Command-R+
    // ══════════════════════════════════════════════════════════════════════════

    /**
     * Cohere Command-R architecture.
     *
     * <p>
     * Distinctive features:
     * - Parallel Attention + FFN (not sequential like LLaMA)
     * - Rotary position embeddings with different base (10000 for R, 2000 for R+)
     * - LayerNorm (not RMSNorm) with bias
     * - No gate projection — uses standard up/down projection
     *
     * <p>
     * Models: c4ai-command-r-v01, c4ai-command-r-plus-v01, command-r-08-2024
     *
     * <p>
     * Weight names:
     * model.layers.{i}.self_attn.q_proj.weight
     * model.layers.{i}.mlp.up_proj.weight (no gate_proj)
     * model.layers.{i}.mlp.down_proj.weight
     * model.layers.{i}.input_layernorm.weight / bias
     */
    @ApplicationScoped
    public static final class CohereCommandFamily implements ModelArchitecture {

        @Override
        public String id() {
            return "cohere";
        }

        @Override
        public List<String> supportedArchClassNames() {
            return List.of("CohereForCausalLM", "Cohere2ForCausalLM");
        }

        @Override
        public List<String> supportedModelTypes() {
            return List.of("cohere", "cohere2");
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
            return "model.embed_tokens.weight";
        } // tied embeddings

        @Override
        public boolean hasTiedEmbeddings() {
            return true;
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

        // Command-R uses standard FFN (no gate projection — SwiGLU becomes just SiLU)
        @Override
        public String layerFfnGateWeight(int i) {
            return null;
        } // no gate

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
            return "model.layers.%d.input_layernorm.weight".formatted(i);
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    // DeepSeek-V3 — Multi-head Latent Attention + MoE
    // ══════════════════════════════════════════════════════════════════════════

    /**
     * DeepSeek-V3 architecture.
     *
     * <p>
     * Key innovations over V2:
     * - Multi-head Latent Attention (MLA): compresses KV cache via low-rank
     * projection (latent_dim << head_dim * num_heads), drastically reducing
     * memory bandwidth at inference
     * - FP8 training weights (shipped as FP8 in HuggingFace checkpoint)
     * - 256 experts total, 8 activated per token
     *
     * <p>
     * MLA weight naming:
     * model.layers.{i}.self_attn.q_proj.weight [num_heads*head_dim, hidden]
     * model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight
     * [kv_lora_rank+rope_head_dim, hidden]
     * model.layers.{i}.self_attn.kv_b_proj.weight [num_heads*(head_dim+v_head_dim),
     * kv_lora_rank]
     * model.layers.{i}.self_attn.o_proj.weight
     *
     * <p>
     * Models: DeepSeek-V3 (671B MoE), DeepSeek-V3-0324
     */
    @ApplicationScoped
    public static final class DeepSeekV3Family implements ModelArchitecture {

        @Override
        public String id() {
            return "deepseek_v3";
        }

        @Override
        public List<String> supportedArchClassNames() {
            return List.of("DeepseekV3ForCausalLM");
        }

        @Override
        public List<String> supportedModelTypes() {
            return List.of("deepseek_v3");
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

        // MLA attention: q_proj is standard, KV uses low-rank compressed projection
        @Override
        public String layerQueryWeight(int i) {
            return "model.layers.%d.self_attn.q_proj.weight".formatted(i);
        }

        @Override
        public String layerKeyWeight(int i) {
            return "model.layers.%d.self_attn.kv_a_proj_with_mqa.weight".formatted(i);
        }

        @Override
        public String layerValueWeight(int i) {
            return "model.layers.%d.self_attn.kv_b_proj.weight".formatted(i);
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

        // MLA-specific weight accessors
        public String kvaQNorm(int i) {
            return "model.layers.%d.self_attn.q_a_layernorm.weight".formatted(i);
        }

        public String kvaKVNorm(int i) {
            return "model.layers.%d.self_attn.kv_a_layernorm.weight".formatted(i);
        }

        public String qAProj(int i) {
            return "model.layers.%d.self_attn.q_a_proj.weight".formatted(i);
        }

        public String qBProj(int i) {
            return "model.layers.%d.self_attn.q_b_proj.weight".formatted(i);
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    // OLMo-2 — Allen AI
    // ══════════════════════════════════════════════════════════════════════════

    /**
     * OLMo-2 (Open Language Model, Allen AI).
     *
     * <p>
     * Post-norm architecture (norm after attention, not before).
     * Uses rotary position embeddings and SwiGLU.
     *
     * <p>
     * Models: OLMo-2-7B, OLMo-2-13B
     */
    @ApplicationScoped
    public static final class OLMo2Family implements ModelArchitecture {

        @Override
        public String id() {
            return "olmo2";
        }

        @Override
        public List<String> supportedArchClassNames() {
            return List.of("OLMo2ForCausalLM", "Olmo2ForCausalLM");
        }

        @Override
        public List<String> supportedModelTypes() {
            return List.of("olmo2");
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

        // OLMo-2: post-attention norm (after, not before)
        @Override
        public String layerAttentionNormWeight(int i) {
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

        @Override
        public String layerFfnNormWeight(int i) {
            return "model.layers.%d.post_feedforward_layernorm.weight".formatted(i);
        }

        // Q/K norms (OLMo-2 applies separate norms to Q and K)
        public String layerQNorm(int i) {
            return "model.layers.%d.self_attn.q_norm.weight".formatted(i);
        }

        public String layerKNorm(int i) {
            return "model.layers.%d.self_attn.k_norm.weight".formatted(i);
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Mistral-NeMo — Mistral × NVIDIA 12B
    // ══════════════════════════════════════════════════════════════════════════

    /**
     * Mistral-NeMo architecture (12B, Mistral AI × NVIDIA).
     *
     * <p>
     * Identical weight layout to Mistral-7B-v0.3 but:
     * - Larger model (12B vs 7B)
     * - Tekken tokenizer (131k vocabulary, BPE)
     * - Sliding window: 4096 tokens
     * - Trained with NeMo framework
     *
     * <p>
     * Models: Mistral-Nemo-Instruct-2407, Mistral-Nemo-Base-2407
     */
    @ApplicationScoped
    public static final class MistralNeMoFamily implements ModelArchitecture {

        @Override
        public String id() {
            return "mistral_nemo";
        }

        @Override
        public List<String> supportedArchClassNames() {
            return List.of("MistralForCausalLM");
        }

        @Override
        public List<String> supportedModelTypes() {
            return List.of("mistral_nemo");
        }

        // Identical naming to Mistral-7B
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

    // ══════════════════════════════════════════════════════════════════════════
    // SmolLM / SmolLM2 — HuggingFace compact models
    // ══════════════════════════════════════════════════════════════════════════

    /**
     * SmolLM-2 architecture (HuggingFace, 135M–1.7B).
     *
     * <p>
     * LLaMA-style architecture at very small scale.
     * Excellent for on-device and edge inference.
     * Models: SmolLM2-135M, SmolLM2-360M, SmolLM2-1.7B
     */
    @ApplicationScoped
    public static final class SmolLMFamily implements ModelArchitecture {

        @Override
        public String id() {
            return "smollm";
        }

        @Override
        public List<String> supportedArchClassNames() {
            return List.of("LlamaForCausalLM");
        }

        @Override
        public List<String> supportedModelTypes() {
            return List.of("smollm", "smollm2");
        }

        // Same as LLaMA — SmolLM uses LlamaForCausalLM arch class
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

    // ══════════════════════════════════════════════════════════════════════════
    // Phi-4 — Microsoft Phi-4-mini and Phi-4
    // ══════════════════════════════════════════════════════════════════════════

    /**
     * Microsoft Phi-4 architecture.
     *
     * <p>
     * Phi-4-mini (3.8B) and Phi-4 (14B).
     * Fused QKV projection, grouped-query attention.
     * Uses a 100K token vocabulary (tiktoken-based).
     *
     * <p>
     * Models: Phi-4, Phi-4-mini, Phi-4-mini-instruct
     */
    @ApplicationScoped
    public static final class Phi4Family implements ModelArchitecture {

        @Override
        public String id() {
            return "phi4";
        }

        @Override
        public List<String> supportedArchClassNames() {
            return List.of("Phi3ForCausalLM", "PhiMoEForCausalLM");
        }

        @Override
        public List<String> supportedModelTypes() {
            return List.of("phi4", "phi4-mini");
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
        public String layerFfnGateWeight(int i) {
            return "model.layers.%d.mlp.gate_up_proj.weight".formatted(i);
        }

        @Override
        public String layerFfnUpWeight(int i) {
            return null;
        } // fused with gate

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
