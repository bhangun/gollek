/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * MultimodalModelFamilies.java
 * ─────────────────────────────
 * ModelArchitecture beans for all major vision-language model families.
 *
 * Each family provides:
 *   1. LLM backbone weight names  (inherited from text-only family)
 *   2. Vision encoder weight names (CLIP / ViT prefix)
 *   3. Projection layer weight names (MLP bridge LLM↔ViT)
 *   4. VisionConfig factory method
 *
 * Supported VLM families
 * ═══════════════════════
 *   InternVL2          — InternLM-2 + InternViT-300M/6B
 *                        Models: InternVL2-1B, 2B, 4B, 8B, 26B, 40B, 76B
 *
 *   Phi-3-Vision       — Phi-3-mini + CLIP-ViT-L/14@336
 *                        Models: Phi-3-vision-128k-instruct
 *
 *   LLaMA-3.2-Vision   — LLaMA-3.1 + custom cross-attention vision adapter
 *                        Models: Llama-3.2-11B-Vision, 90B-Vision
 *
 *   Gemma3             — Gemma-3 with native multimodal support
 *                        Models: gemma-3-4b-it, gemma-3-12b-it, gemma-3-27b-it
 *
 *   Qwen2-VL           — Qwen2 + ViT with native resolution (NaViT)
 *                        Models: Qwen2-VL-2B, 7B, 72B-Instruct
 *
 *   DeepSeek-VL2       — DeepSeek-MoE + SigLIP vision encoder
 *                        Models: DeepSeek-VL2-Tiny, Small, Base, Pro
 *
 *   Molmo              — OLMo + CLIP (Allen AI)
 *                        Models: Molmo-7B-D, Molmo-72B
 */
package tech.kayys.gollek.safetensor.vision;

import jakarta.enterprise.context.ApplicationScoped;
import tech.kayys.gollek.safetensor.vision.VisionEncoder.VisionConfig;
import tech.kayys.gollek.spi.model.ModelArchitecture;

import java.util.List;

public final class MultimodalModelFamilies {

    private MultimodalModelFamilies() {
    }

    // ══════════════════════════════════════════════════════════════════════════
    // InternVL2 — InternLM-2 + InternViT
    // ══════════════════════════════════════════════════════════════════════════

    /**
     * InternVL2 architecture.
     *
     * <p>
     * Weight layout:
     * 
     * <pre>
     *   vision_model.encoder.layers.{i}.attn.qkv.weight   [3*dim, dim]
     *   vision_model.encoder.layers.{i}.mlp.fc1.weight
     *   vision_model.encoder.layers.{i}.mlp.fc2.weight
     *   vision_model.encoder.layers.{i}.norm1/norm2.weight/bias
     *   mlp1.0.weight  — pixel shuffle projection [4*vDim, llmDim]
     *   mlp1.2.weight  — projection layer 2
     *   language_model.model.embed_tokens.weight
     *   language_model.model.layers.{i}.self_attn.q_proj.weight
     * </pre>
     */
    @ApplicationScoped
    public static final class InternVL2Family implements ModelArchitecture {

        @Override
        public String id() {
            return "internvl2";
        }

        @Override
        public List<String> supportedArchClassNames() {
            return List.of("InternVLChatModel", "InternVL2ForCausalLM");
        }

        @Override
        public List<String> supportedModelTypes() {
            return List.of("internvl_chat", "internvl2");
        }

        // ── LLM backbone (InternLM-2 style, nested under language_model) ────

        @Override
        public String embedTokensWeight() {
            return "language_model.model.embed_tokens.weight";
        }

        @Override
        public String finalNormWeight() {
            return "language_model.model.norm.weight";
        }

        @Override
        public String lmHeadWeight() {
            return "language_model.lm_head.weight";
        }

        @Override
        public String layerQueryWeight(int i) {
            return "language_model.model.layers.%d.self_attn.q_proj.weight".formatted(i);
        }

        @Override
        public String layerKeyWeight(int i) {
            return "language_model.model.layers.%d.self_attn.k_proj.weight".formatted(i);
        }

        @Override
        public String layerValueWeight(int i) {
            return "language_model.model.layers.%d.self_attn.v_proj.weight".formatted(i);
        }

        @Override
        public String layerOutputWeight(int i) {
            return "language_model.model.layers.%d.self_attn.o_proj.weight".formatted(i);
        }

        @Override
        public String layerAttentionNormWeight(int i) {
            return "language_model.model.layers.%d.input_layernorm.weight".formatted(i);
        }

        @Override
        public String layerFfnGateWeight(int i) {
            return "language_model.model.layers.%d.mlp.gate_proj.weight".formatted(i);
        }

        @Override
        public String layerFfnUpWeight(int i) {
            return "language_model.model.layers.%d.mlp.up_proj.weight".formatted(i);
        }

        @Override
        public String layerFfnDownWeight(int i) {
            return "language_model.model.layers.%d.mlp.down_proj.weight".formatted(i);
        }

        @Override
        public String layerFfnNormWeight(int i) {
            return "language_model.model.layers.%d.post_attention_layernorm.weight".formatted(i);
        }

        // ── Vision-specific weight accessors ─────────────────────────────────

        public String visionLayerQKV(int i) {
            return "vision_model.encoder.layers.%d.attn.qkv.weight".formatted(i);
        }

        public String visionLayerOut(int i) {
            return "vision_model.encoder.layers.%d.attn.proj.weight".formatted(i);
        }

        public String visionLayerNorm1W(int i) {
            return "vision_model.encoder.layers.%d.norm1.weight".formatted(i);
        }

        public String visionLayerNorm2W(int i) {
            return "vision_model.encoder.layers.%d.norm2.weight".formatted(i);
        }

        public String visionLayerMlpFc1(int i) {
            return "vision_model.encoder.layers.%d.mlp.fc1.weight".formatted(i);
        }

        public String visionLayerMlpFc2(int i) {
            return "vision_model.encoder.layers.%d.mlp.fc2.weight".formatted(i);
        }

        public String visionProjLayer1() {
            return "mlp1.0.weight";
        }

        public String visionProjLayer2() {
            return "mlp1.2.weight";
        }

        /** InternViT-300M config (used in InternVL2-1B, 2B). */
        public VisionConfig visionConfig300M(int llmDim) {
            return new VisionConfig(448, 14, 1024, 24, 16, llmDim, "vision_model.encoder.layers.");
        }

        /** InternViT-6B config (used in InternVL2-8B, 26B, 76B). */
        public VisionConfig visionConfig6B(int llmDim) {
            return new VisionConfig(448, 14, 3200, 45, 25, llmDim, "vision_model.encoder.layers.");
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Phi-3-Vision — Phi-3-mini + CLIP-ViT-L/14
    // ══════════════════════════════════════════════════════════════════════════

    /**
     * Phi-3-Vision architecture (phi-3-vision-128k-instruct).
     *
     * <p>
     * LLM weights: standard Phi-3 naming.
     * Vision weights: {@code model.vision_embed_tokens.*}
     * Projection: {@code model.vision_embed_tokens.img_projection.*}
     */
    @ApplicationScoped
    public static final class Phi3VisionFamily implements ModelArchitecture {

        @Override
        public String id() {
            return "phi3_v";
        }

        @Override
        public List<String> supportedArchClassNames() {
            return List.of("Phi3VForCausalLM", "Phi3VisionForCausalLM");
        }

        @Override
        public List<String> supportedModelTypes() {
            return List.of("phi3_v", "phi-3-vision-128k-instruct");
        }

        // ── LLM backbone (Phi-3 style) ────────────────────────────────────────
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
            return "model.layers.%d.self_attn.qkv_proj.weight".formatted(i);
        }

        @Override
        public String layerKeyWeight(int i) {
            return null;
        } // fused QKV

        @Override
        public String layerValueWeight(int i) {
            return null;
        } // fused QKV

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
        } // fused gate+up

        @Override
        public String layerFfnDownWeight(int i) {
            return "model.layers.%d.mlp.down_proj.weight".formatted(i);
        }

        @Override
        public String layerFfnNormWeight(int i) {
            return "model.layers.%d.post_attention_layernorm.weight".formatted(i);
        }

        @Override
        public boolean hasFusedQKV() {
            return true;
        }

        @Override
        public String layerFusedQKVWeight(int i) {
            return "model.layers.%d.self_attn.qkv_proj.weight".formatted(i);
        }

        // ── Vision-specific ───────────────────────────────────────────────────
        public String visionEmbedPrefix() {
            return "model.vision_embed_tokens.";
        }

        public String visionImgProjection() {
            return "model.vision_embed_tokens.img_projection.0.weight";
        }

        public VisionConfig visionConfig(int llmDim) {
            return new VisionConfig(336, 14, 1024, 24, 16, llmDim,
                    "model.vision_embed_tokens.img_processor.vision_model.encoder.layers.");
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    // LLaMA-3.2-Vision — LLaMA-3.1 + cross-attention vision adapter
    // ══════════════════════════════════════════════════════════════════════════

    /**
     * LLaMA-3.2-Vision architecture (Llama-3.2-11B-Vision-Instruct).
     *
     * <p>
     * Unique: uses cross-attention adapters interleaved with LLaMA-3 decoder
     * layers instead of concatenating image tokens with text tokens.
     *
     * <p>
     * Vision encoder: {@code vision_model.*} (CLIP-style)
     * Projection: {@code multi_modal_projector.*}
     * Cross-attention: {@code language_model.model.layers.{i}.cross_attn.*}
     * (only on layers 3, 8, 13, 18, 23, 28, 33, 38)
     */
    @ApplicationScoped
    public static final class LlamaVisionFamily implements ModelArchitecture {

        @Override
        public String id() {
            return "mllama";
        }

        @Override
        public List<String> supportedArchClassNames() {
            return List.of("MllamaForConditionalGeneration");
        }

        @Override
        public List<String> supportedModelTypes() {
            return List.of("mllama");
        }

        // ── LLM backbone (LLaMA-3.1 under language_model.*) ─────────────────
        @Override
        public String embedTokensWeight() {
            return "language_model.model.embed_tokens.weight";
        }

        @Override
        public String finalNormWeight() {
            return "language_model.model.norm.weight";
        }

        @Override
        public String lmHeadWeight() {
            return "language_model.lm_head.weight";
        }

        @Override
        public String layerQueryWeight(int i) {
            return "language_model.model.layers.%d.self_attn.q_proj.weight".formatted(i);
        }

        @Override
        public String layerKeyWeight(int i) {
            return "language_model.model.layers.%d.self_attn.k_proj.weight".formatted(i);
        }

        @Override
        public String layerValueWeight(int i) {
            return "language_model.model.layers.%d.self_attn.v_proj.weight".formatted(i);
        }

        @Override
        public String layerOutputWeight(int i) {
            return "language_model.model.layers.%d.self_attn.o_proj.weight".formatted(i);
        }

        @Override
        public String layerAttentionNormWeight(int i) {
            return "language_model.model.layers.%d.input_layernorm.weight".formatted(i);
        }

        @Override
        public String layerFfnGateWeight(int i) {
            return "language_model.model.layers.%d.mlp.gate_proj.weight".formatted(i);
        }

        @Override
        public String layerFfnUpWeight(int i) {
            return "language_model.model.layers.%d.mlp.up_proj.weight".formatted(i);
        }

        @Override
        public String layerFfnDownWeight(int i) {
            return "language_model.model.layers.%d.mlp.down_proj.weight".formatted(i);
        }

        @Override
        public String layerFfnNormWeight(int i) {
            return "language_model.model.layers.%d.post_attention_layernorm.weight".formatted(i);
        }

        // ── Vision cross-attention layers ─────────────────────────────────────

        /** Cross-attention query weight (only on vision-adapter layers). */
        public String crossAttnQWeight(int i) {
            return "language_model.model.layers.%d.cross_attn.q_proj.weight".formatted(i);
        }

        public String crossAttnKWeight(int i) {
            return "language_model.model.layers.%d.cross_attn.k_proj.weight".formatted(i);
        }

        public String crossAttnVWeight(int i) {
            return "language_model.model.layers.%d.cross_attn.v_proj.weight".formatted(i);
        }

        public String crossAttnOWeight(int i) {
            return "language_model.model.layers.%d.cross_attn.o_proj.weight".formatted(i);
        }

        /**
         * Cross-attention adapter layers in LLaMA-3.2-11B (every 4th starting at 3).
         */
        public static final int[] CROSS_ATTN_LAYERS = { 3, 8, 13, 18, 23, 28, 33, 38 };

        public boolean isCrossAttnLayer(int i) {
            for (int ca : CROSS_ATTN_LAYERS)
                if (ca == i)
                    return true;
            return false;
        }

        public VisionConfig visionConfig(int llmDim) {
            return new VisionConfig(560, 14, 1280, 32, 16, llmDim, "vision_model.encoder.layers.");
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Gemma3 — native multimodal (Google)
    // ══════════════════════════════════════════════════════════════════════════

    /**
     * Gemma-3 multimodal architecture.
     *
     * <p>
     * Unique features vs Gemma-2:
     * - SigLIP vision encoder (not CLIP)
     * - Bidirectional sliding window attention in text model
     * - Support for 1:1 / 16:9 / 4:3 aspect ratios without forced square resize
     * - Vision tokens interleaved with text tokens (no cross-attention)
     *
     * <p>
     * Weight names: {@code model.vision_tower.*} for vision,
     * {@code model.multi_modal_projector.*} for projection,
     * {@code model.language_model.*} for LLM backbone.
     */
    @ApplicationScoped
    public static final class Gemma3Family implements ModelArchitecture {

        @Override
        public String id() {
            return "gemma3";
        }

        @Override
        public List<String> supportedArchClassNames() {
            return List.of("Gemma3ForConditionalGeneration", "Gemma3ForCausalLM");
        }

        @Override
        public List<String> supportedModelTypes() {
            return List.of("gemma3");
        }

        // ── LLM (Gemma-3 normalisation differs from Gemma-2) ─────────────────
        @Override
        public String embedTokensWeight() {
            return "model.language_model.embed_tokens.weight";
        }

        @Override
        public String finalNormWeight() {
            return "model.language_model.norm.weight";
        }

        @Override
        public String lmHeadWeight() {
            return "model.language_model.embed_tokens.weight";
        } // tied

        @Override
        public boolean hasTiedEmbeddings() {
            return true;
        }

        @Override
        public String layerQueryWeight(int i) {
            return "model.language_model.layers.%d.self_attn.q_proj.weight".formatted(i);
        }

        @Override
        public String layerKeyWeight(int i) {
            return "model.language_model.layers.%d.self_attn.k_proj.weight".formatted(i);
        }

        @Override
        public String layerValueWeight(int i) {
            return "model.language_model.layers.%d.self_attn.v_proj.weight".formatted(i);
        }

        @Override
        public String layerOutputWeight(int i) {
            return "model.language_model.layers.%d.self_attn.o_proj.weight".formatted(i);
        }

        @Override
        public String layerAttentionNormWeight(int i) {
            return "model.language_model.layers.%d.input_layernorm.weight".formatted(i);
        }

        @Override
        public String layerFfnGateWeight(int i) {
            return "model.language_model.layers.%d.mlp.gate_proj.weight".formatted(i);
        }

        @Override
        public String layerFfnUpWeight(int i) {
            return "model.language_model.layers.%d.mlp.up_proj.weight".formatted(i);
        }

        @Override
        public String layerFfnDownWeight(int i) {
            return "model.language_model.layers.%d.mlp.down_proj.weight".formatted(i);
        }

        @Override
        public String layerFfnNormWeight(int i) {
            return "model.language_model.layers.%d.post_feedforward_layernorm.weight".formatted(i);
        }

        // ── SigLIP vision encoder ─────────────────────────────────────────────
        public String visionWeightPrefix() {
            return "model.vision_tower.vision_model.encoder.layers.";
        }

        public String visionProjW() {
            return "model.multi_modal_projector.mm_soft_emb_norm.weight";
        }

        public String visionLinearProj() {
            return "model.multi_modal_projector.linear.weight";
        }

        /** SigLIP config (ViT-SO400M, 224×224, patch 14). */
        public VisionConfig visionConfig(int llmDim) {
            return new VisionConfig(224, 14, 1152, 27, 16, llmDim, "model.vision_tower.vision_model.encoder.layers.");
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Qwen2-VL — native resolution ViT (NaViT)
    // ══════════════════════════════════════════════════════════════════════════

    /**
     * Qwen2-VL architecture with native dynamic resolution.
     *
     * <p>
     * Unique: 2D-RoPE in the vision encoder (Rotary Position Embedding
     * in both H and W dimensions, not 1D positional embeddings).
     * Processes images at their native aspect ratio without forced square resize.
     *
     * <p>
     * Vision encoder: {@code visual.*}
     * LLM: standard Qwen-2 naming under {@code model.*}
     */
    @ApplicationScoped
    public static final class Qwen2VLFamily implements ModelArchitecture {

        @Override
        public String id() {
            return "qwen2_vl";
        }

        @Override
        public List<String> supportedArchClassNames() {
            return List.of("Qwen2VLForConditionalGeneration");
        }

        @Override
        public List<String> supportedModelTypes() {
            return List.of("qwen2_vl");
        }

        // ── LLM backbone (Qwen-2 style) ───────────────────────────────────────
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

        // ── Vision encoder (ViT with 2D-RoPE) ────────────────────────────────
        public String visualPatchEmbedW() {
            return "visual.patch_embed.proj.weight";
        }

        public String visualLayerQKV(int i) {
            return "visual.blocks.%d.attn.qkv.weight".formatted(i);
        }

        public String visualLayerProj(int i) {
            return "visual.blocks.%d.attn.proj.weight".formatted(i);
        }

        public String visualLayerNorm1W(int i) {
            return "visual.blocks.%d.norm1.weight".formatted(i);
        }

        public String visualLayerNorm2W(int i) {
            return "visual.blocks.%d.norm2.weight".formatted(i);
        }

        public String visualLayerMlpFc1(int i) {
            return "visual.blocks.%d.mlp.fc1.weight".formatted(i);
        }

        public String visualLayerMlpFc2(int i) {
            return "visual.blocks.%d.mlp.fc2.weight".formatted(i);
        }

        public String visualMerger() {
            return "visual.merger.mlp.0.weight";
        }

        /**
         * Qwen2-VL 7B vision config (native resolution, max 28×28 patches = 1568
         * tokens).
         */
        public VisionConfig visionConfig(int llmDim) {
            return new VisionConfig(448, 14, 1664, 32, 16, llmDim, "visual.blocks.");
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    // DeepSeek-VL2 — DeepSeek-MoE + SigLIP
    // ══════════════════════════════════════════════════════════════════════════

    /**
     * DeepSeek-VL2 architecture.
     *
     * <p>
     * Combines DeepSeek-MoE LLM with SigLIP-L vision encoder.
     * Unique: tile-based high-resolution encoding and dynamic tiling.
     *
     * <p>
     * Models: DeepSeek-VL2-Tiny (3B), Small (16B), Base (27B), Pro (72B)
     */
    @ApplicationScoped
    public static final class DeepSeekVL2Family implements ModelArchitecture {

        @Override
        public String id() {
            return "deepseek_vl_v2";
        }

        @Override
        public List<String> supportedArchClassNames() {
            return List.of("DeepseekVL2ForCausalLM");
        }

        @Override
        public List<String> supportedModelTypes() {
            return List.of("deepseek_vl_v2");
        }

        // ── LLM backbone (DeepSeek-MoE style) ────────────────────────────────
        @Override
        public String embedTokensWeight() {
            return "language_model.model.embed_tokens.weight";
        }

        @Override
        public String finalNormWeight() {
            return "language_model.model.norm.weight";
        }

        @Override
        public String lmHeadWeight() {
            return "language_model.lm_head.weight";
        }

        @Override
        public String layerQueryWeight(int i) {
            return "language_model.model.layers.%d.self_attn.q_proj.weight".formatted(i);
        }

        @Override
        public String layerKeyWeight(int i) {
            return "language_model.model.layers.%d.self_attn.k_proj.weight".formatted(i);
        }

        @Override
        public String layerValueWeight(int i) {
            return "language_model.model.layers.%d.self_attn.v_proj.weight".formatted(i);
        }

        @Override
        public String layerOutputWeight(int i) {
            return "language_model.model.layers.%d.self_attn.o_proj.weight".formatted(i);
        }

        @Override
        public String layerAttentionNormWeight(int i) {
            return "language_model.model.layers.%d.input_layernorm.weight".formatted(i);
        }

        @Override
        public String layerFfnGateWeight(int i) {
            return "language_model.model.layers.%d.mlp.gate_proj.weight".formatted(i);
        }

        @Override
        public String layerFfnUpWeight(int i) {
            return "language_model.model.layers.%d.mlp.up_proj.weight".formatted(i);
        }

        @Override
        public String layerFfnDownWeight(int i) {
            return "language_model.model.layers.%d.mlp.down_proj.weight".formatted(i);
        }

        @Override
        public String layerFfnNormWeight(int i) {
            return "language_model.model.layers.%d.post_attention_layernorm.weight".formatted(i);
        }

        // ── SigLIP vision encoder ─────────────────────────────────────────────
        public String visionWeightPrefix() {
            return "vision_tower.model.vision_tower.vision_model.encoder.layers.";
        }

        public String visionProjection1() {
            return "aligner.fc1.weight";
        }

        public String visionProjection2() {
            return "aligner.fc2.weight";
        }

        public VisionConfig visionConfig(int llmDim) {
            return new VisionConfig(384, 16, 1024, 27, 16, llmDim,
                    "vision_tower.model.vision_tower.vision_model.encoder.layers.");
        }
    }

    // ══════════════════════════════════════════════════════════════════════════
    // Molmo — OLMo + CLIP (Allen AI)
    // ══════════════════════════════════════════════════════════════════════════

    /**
     * Molmo architecture (Allen AI).
     *
     * <p>
     * CLIP-L + OLMo-7B backbone.
     * Unique: multiple CLIP crops at different scales (multi-scale encoding).
     * Models: Molmo-7B-D-0924, Molmo-72B-0924
     */
    @ApplicationScoped
    public static final class MolmoFamily implements ModelArchitecture {

        @Override
        public String id() {
            return "molmo";
        }

        @Override
        public List<String> supportedArchClassNames() {
            return List.of("MolmoForCausalLM");
        }

        @Override
        public List<String> supportedModelTypes() {
            return List.of("molmo");
        }

        // ── LLM backbone (OLMo / LLaMA-style) ────────────────────────────────
        @Override
        public String embedTokensWeight() {
            return "model.transformer.wte.weight";
        }

        @Override
        public String finalNormWeight() {
            return "model.transformer.ln_f.weight";
        }

        @Override
        public String lmHeadWeight() {
            return "model.transformer.ff_out.weight";
        }

        @Override
        public String layerQueryWeight(int i) {
            return "model.transformer.blocks.%d.att_proj.weight".formatted(i);
        }

        @Override
        public String layerKeyWeight(int i) {
            return null;
        } // fused QKV

        @Override
        public String layerValueWeight(int i) {
            return null;
        }

        @Override
        public boolean hasFusedQKV() {
            return true;
        }

        @Override
        public String layerFusedQKVWeight(int i) {
            return "model.transformer.blocks.%d.att_proj.weight".formatted(i);
        }

        @Override
        public String layerOutputWeight(int i) {
            return "model.transformer.blocks.%d.attn_out.weight".formatted(i);
        }

        @Override
        public String layerAttentionNormWeight(int i) {
            return "model.transformer.blocks.%d.attn_norm.weight".formatted(i);
        }

        @Override
        public String layerFfnGateWeight(int i) {
            return "model.transformer.blocks.%d.ff_proj.weight".formatted(i);
        }

        @Override
        public String layerFfnUpWeight(int i) {
            return null;
        } // fused gate+up

        @Override
        public String layerFfnDownWeight(int i) {
            return "model.transformer.blocks.%d.ff_out.weight".formatted(i);
        }

        @Override
        public String layerFfnNormWeight(int i) {
            return "model.transformer.blocks.%d.ff_norm.weight".formatted(i);
        }

        // ── CLIP vision encoder ───────────────────────────────────────────────
        public String visionWeightPrefix() {
            return "model.vision_backbone.image_vit.encoder.layers.";
        }

        public String imageProjectorW() {
            return "model.vision_backbone.image_projector.weight";
        }

        public VisionConfig visionConfig(int llmDim) {
            return new VisionConfig(336, 14, 1024, 24, 16, llmDim,
                    "model.vision_backbone.image_vit.encoder.layers.");
        }
    }
}
