/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * VisionModelRegistry.java
 * ─────────────────────────
 * Central registry that maps model type / architecture class to the
 * correct VisionConfig and vision encoder weight names.
 *
 * Used by MultimodalInferenceEngine to avoid hard-coding the vision
 * config in caller code.
 *
 * Registration
 * ════════════
 * All VLM architectures register themselves at startup via CDI.
 * MultimodalInferenceEngine calls resolve(modelType, llmDim) to get
 * the vision config for a specific model.
 */
package tech.kayys.gollek.safetensor.vision;

import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;

import org.jboss.logging.Logger;

import java.util.*;

/**
 * Resolves vision configuration for a given LLM architecture.
 */
@ApplicationScoped
public class VisionModelRegistry {

    private static final Logger log = Logger.getLogger(VisionModelRegistry.class);

    @Inject
    MultimodalModelFamilies.InternVL2Family internVl2;
    @Inject
    MultimodalModelFamilies.Phi3VisionFamily phi3Vision;
    @Inject
    MultimodalModelFamilies.LlamaVisionFamily llamaVision;
    @Inject
    MultimodalModelFamilies.Gemma3Family gemma3;
    @Inject
    MultimodalModelFamilies.Qwen2VLFamily qwen2Vl;
    @Inject
    MultimodalModelFamilies.DeepSeekVL2Family deepSeekVl2;
    @Inject
    MultimodalModelFamilies.MolmoFamily molmo;

    /**
     * Look up whether a model type has a vision encoder.
     */
    public boolean isMultimodal(String modelType) {
        if (modelType == null)
            return false;
        return switch (modelType.toLowerCase()) {
            case "internvl_chat", "internvl2",
                    "phi3_v", "phi-3-vision-128k-instruct",
                    "mllama", "llama-3.2-vision",
                    "gemma3",
                    "qwen2_vl",
                    "deepseek_vl_v2",
                    "molmo",
                    "llava", "llava_next",
                    "idefics2", "idefics3" ->
                true;
            default -> false;
        };
    }

    /**
     * Resolve the vision configuration for a given model type and LLM dimension.
     *
     * @param modelType model_type from config.json
     * @param llmDim    LLM hidden_size (used to size the projection layer)
     * @return VisionConfig with correct image size, patch size, ViT dimension
     */
    public VisionEncoder.VisionConfig resolve(String modelType, int llmDim) {
        if (modelType == null)
            return VisionEncoder.VisionConfig.llava15(llmDim);
        return switch (modelType.toLowerCase()) {
            case "internvl_chat", "internvl2" -> internVl2.visionConfig300M(llmDim);
            case "phi3_v" -> phi3Vision.visionConfig(llmDim);
            case "mllama" -> llamaVision.visionConfig(llmDim);
            case "gemma3" -> gemma3.visionConfig(llmDim);
            case "qwen2_vl" -> qwen2Vl.visionConfig(llmDim);
            case "deepseek_vl_v2" -> deepSeekVl2.visionConfig(llmDim);
            case "molmo" -> molmo.visionConfig(llmDim);
            case "qwen_vl" -> VisionEncoder.VisionConfig.qwenVL(llmDim);
            default -> {
                log.debugf("VisionModelRegistry: unknown VLM type '%s' — using LLaVA-1.5 defaults", modelType);
                yield VisionEncoder.VisionConfig.llava15(llmDim);
            }
        };
    }

    /**
     * Resolve the VLM prompt template for a model type.
     * Returns the formatted prompt with the image placeholder and user text.
     */
    public String buildPrompt(String modelType, String userText) {
        if (modelType == null)
            modelType = "llava";
        return switch (modelType.toLowerCase()) {
            case "qwen2_vl", "qwen_vl" ->
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                        + "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
                        + userText + "<|im_end|>\n<|im_start|>assistant\n";

            case "mllama" ->
                "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                        + "<|image|>" + userText + "<|eot_id|>"
                        + "<|start_header_id|>assistant<|end_header_id|>\n\n";

            case "phi3_v", "phi-3-vision-128k-instruct" ->
                "<|user|>\n<|image_1|>\n" + userText + "<|end|>\n<|assistant|>\n";

            case "gemma3" ->
                "<start_of_turn>user\n<image>\n" + userText + "<end_of_turn>\n"
                        + "<start_of_turn>model\n";

            case "internvl_chat", "internvl2" ->
                "<|im_start|>system\n你是书生·万象，英文名是InternVL，是由上海人工智能实验室及多家合作单位联合开发的多模态大语言模型。<|im_end|>"
                        + "<|im_start|>user\n<image>\n" + userText + "<|im_end|>\n"
                        + "<|im_start|>assistant\n";

            case "deepseek_vl_v2" ->
                "<|ref|><image><|/ref|>" + userText;

            case "molmo" ->
                " User: <image>\n" + userText + " Assistant:";

            // LLaVA / default
            default ->
                "USER: <image>\n" + userText + " ASSISTANT:";
        };
    }

    /**
     * All registered multimodal model types (for health/info endpoints).
     */
    public List<String> registeredTypes() {
        return List.of(
                "internvl2", "phi3_v", "mllama", "gemma3",
                "qwen2_vl", "deepseek_vl_v2", "molmo",
                "llava", "llava_next", "qwen_vl");
    }
}
