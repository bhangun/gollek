/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * ModelArchitectureDetectorV2.java
 * ─────────────────────────────────
 * Heuristic detection of model architecture from weight tensor names,
 * config.json fields, and file patterns. Used as a fallback when
 * config.json doesn't contain a recognized model_type.
 *
 * Detection order (highest to lowest confidence):
 *   1. Exact model_type match (definitive)
 *   2. Architecture class name match (definitive)
 *   3. TorchTensor name signature patterns (heuristic)
 *   4. File-level heuristics (model card, README, tokenizer)
 *
 * Covers all 30+ families registered in the system.
 */
package tech.kayys.gollek.safetensor.vision;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;

import java.util.*;
import java.util.stream.Collectors;

import tech.kayys.gollek.safetensor.models.ModelArchitectureRegistry;
import tech.kayys.gollek.spi.model.ModelArchitecture;

/**
 * Detects model architecture from tensor name patterns when config.json
 * doesn't declare a recognized model_type.
 */
@ApplicationScoped
public class ModelArchitectureDetectorV2 {

    private static final Logger log = Logger.getLogger(ModelArchitectureDetectorV2.class);

    @Inject
    ModelArchitectureRegistry registry;

    // ─────────────────────────────────────────────────────────────────────────
    // Detection rules — ordered by specificity
    // ─────────────────────────────────────────────────────────────────────────

    /** TorchTensor name prefix → architecture ID (most specific first). */
    private static final List<DetectionRule> TENSOR_RULES = List.of(
            // Multimodal
            new DetectionRule("vision_model.encoder.layers.0.attn.qkv", "internvl2"),
            new DetectionRule("model.vision_embed_tokens.img_projection", "phi3_v"),
            new DetectionRule("language_model.model.layers.0.cross_attn", "mllama"),
            new DetectionRule("model.language_model.layers.0.self_attn", "gemma3"),
            new DetectionRule("visual.blocks.0.attn.qkv", "qwen2_vl"),
            new DetectionRule("vision_tower.model.vision_tower", "deepseek_vl_v2"),
            new DetectionRule("model.vision_backbone.image_vit", "molmo"),
            new DetectionRule("text_model.embeddings.token_embedding", "clip"),

            // MoE
            new DetectionRule("model.layers.0.block_sparse_moe.gate", "mixtral"),
            new DetectionRule("model.layers.0.mlp.experts.0.gate_proj", "deepseek_v3"),
            new DetectionRule("language_model.model.layers.0.mlp.experts", "deepseek_vl_v2"),

            // Text architectures
            new DetectionRule("model.layers.0.self_attn.kv_a_proj_with_mqa", "deepseek_v3"),
            new DetectionRule("language_model.model.embed_tokens", "internvl2"),
            new DetectionRule("model.transformer.wte", "molmo"),
            new DetectionRule("model.transformer.blocks.0.att_proj", "molmo"),
            new DetectionRule("transformer.visual.transformer", "qwen_vl"),
            new DetectionRule("model.layers.0.self_attn.qkv_proj", "phi3"),
            new DetectionRule("model.layers.0.mlp.gate_up_proj", "phi4"),
            new DetectionRule("model.layers.0.post_attention_layernorm", "olmo2"),
            new DetectionRule("model.transformer.blocks.0.ff_proj", "olmo2"),
            new DetectionRule("vision_tower.vision_model", "llava"),
            new DetectionRule("model.layers.0.self_attn.k_proj", "llama"),
            new DetectionRule("transformer.h.0.self_attention", "falcon"),
            new DetectionRule("gpt_neox.layers.0", "gptneox"),
            new DetectionRule("transformer.encoder.layers.0", "chatglm"),
            new DetectionRule("model.layers.0.attention.wqkv", "internlm"),
            new DetectionRule("model.layers.0.post_feedforward_layernorm", "gemma2"),
            new DetectionRule("model.embed_tokens", "llama"));

    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Detect architecture from a set of tensor names.
     *
     * @param tensorNames collection of all weight tensor names in the checkpoint
     * @return detected architecture ID, or "llama" as default
     */
    public String detectFromTensorNames(Collection<String> tensorNames) {
        if (tensorNames == null || tensorNames.isEmpty())
            return "llama";

        // Build a quick lookup set for prefix checking
        Set<String> names = Set.copyOf(tensorNames);

        for (DetectionRule rule : TENSOR_RULES) {
            for (String name : names) {
                if (name.startsWith(rule.prefix())) {
                    log.infof("ModelArchitectureDetector: detected '%s' from tensor '%s'",
                            rule.archId(), name);
                    return rule.archId();
                }
            }
        }

        log.debugf("ModelArchitectureDetector: no specific match found — defaulting to 'llama'");
        return "llama";
    }

    /**
     * Detect from config.json architectures list.
     *
     * @param archClassNames list of architecture class names (e.g.
     *                       ["LlamaForCausalLM"])
     * @return detected architecture ID, or null if not found
     */
    public Optional<String> detectFromArchClass(List<String> archClassNames) {
        if (archClassNames == null || archClassNames.isEmpty())
            return Optional.empty();
        for (ModelArchitecture arch : registry.getAllArchitectures()) {
            for (String cls : archClassNames) {
                if (arch.supportedArchClassNames().contains(cls)) {
                    log.infof("ModelArchitectureDetector: matched '%s' via arch class '%s'",
                            arch.id(), cls);
                    return Optional.of(arch.id());
                }
            }
        }
        return Optional.empty();
    }

    /**
     * Detect from model_type string (exact match preferred).
     */
    public Optional<String> detectFromModelType(String modelType) {
        if (modelType == null || modelType.isBlank())
            return Optional.empty();
        for (ModelArchitecture arch : registry.getAllArchitectures()) {
            if (arch.supportedModelTypes().contains(modelType)) {
                return Optional.of(arch.id());
            }
        }
        return Optional.empty();
    }

    /**
     * Full detection pipeline: tries model_type → arch class → tensor names.
     *
     * @param modelType   from config.json (may be null)
     * @param archClasses from config.json architectures list (may be empty)
     * @param tensorNames all weight tensor names in the checkpoint
     * @return resolved architecture ID
     */
    public String detect(String modelType, List<String> archClasses,
            Collection<String> tensorNames) {
        // 1. Exact model_type match
        Optional<String> fromType = detectFromModelType(modelType);
        if (fromType.isPresent())
            return fromType.get();

        // 2. Architecture class match
        Optional<String> fromClass = detectFromArchClass(archClasses);
        if (fromClass.isPresent())
            return fromClass.get();

        // 3. TorchTensor name heuristics
        return detectFromTensorNames(tensorNames);
    }

    // ─────────────────────────────────────────────────────────────────────────

    private record DetectionRule(String prefix, String archId) {
    }
}
