/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * ModelConfig.java
 * ─────────────────
 * Model configuration from config.json.
 */
package tech.kayys.gollek.spi.model;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonSetter;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.JsonNode;
import org.jboss.logging.Logger;

/**
 * Parsed HuggingFace {@code config.json} for a transformer model checkpoint.
 *
 * <p>
 * Load via {@link #load(Path, ObjectMapper)} or
 * {@link #fromDirectory(Path, ObjectMapper)}.
 *
 * <p>
 * All fields default to sensible values if absent from the JSON — this
 * handles both slim community configs and full Transformers library exports.
 */
@JsonIgnoreProperties(ignoreUnknown = true)
public class ModelConfig {

    public ModelConfig() {
        // Defaults applied via field initializers
    }

    private static final Logger log = Logger.getLogger(ModelConfig.class);

    // ── Architecture identity ─────────────────────────────────────────────────

    /**
     * Architecture type string — maps to a registered {@link ModelArchitecture}.
     * Examples: "LlamaForCausalLM", "MistralForCausalLM", "GemmaForCausalLM",
     * "Qwen2ForCausalLM", "PhiForCausalLM", "GPTNeoXForCausalLM".
     */
    @JsonProperty("architectures")
    private List<String> architectures;

    /** Short model-type key: "llama", "mistral", "gemma", "phi", "qwen2", etc. */
    @JsonProperty("model_type")
    private String modelType;

    // ── Core dimensions ───────────────────────────────────────────────────────

    /** Embedding / residual stream dimension. */
    @JsonProperty("hidden_size")
    private int hiddenSize = 4096;

    /** Number of transformer layers. */
    @JsonProperty("num_hidden_layers")
    private int numHiddenLayers = 32;

    /** Number of attention heads (query heads). */
    @JsonProperty("num_attention_heads")
    private int numAttentionHeads = 32;

    /**
     * Number of key-value heads.
     * Equal to {@code numAttentionHeads} for MHA.
     * Less than {@code numAttentionHeads} for GQA (LLaMA-2-70B, Mistral, Gemma2).
     * 1 for MQA.
     */
    @JsonProperty("num_key_value_heads")
    private Integer numKeyValueHeads; // null → defaults to numAttentionHeads

    /** FFN intermediate dimension. Typically 4× or 2.67× hidden_size. */
    @JsonProperty("intermediate_size")
    private int intermediateSize = 11008;

    /** Vocabulary size. */
    @JsonProperty("vocab_size")
    private int vocabSize = 32000;

    /** Maximum sequence length the model was trained on. */
    @JsonProperty("max_position_embeddings")
    private int maxPositionEmbeddings = 4096;

    /** Hidden size for per-layer input pathway (Gemma-4 style). */
    @JsonProperty("hidden_size_per_layer_input")
    private int hiddenSizePerLayerInput = 0;

    /** Vocab size for per-layer input pathway (Gemma-4 style). */
    @JsonProperty("vocab_size_per_layer_input")
    private int vocabSizePerLayerInput = 0;

    // ── Normalization ─────────────────────────────────────────────────────────

    /** RMS norm epsilon (LLaMA, Gemma, Mistral). */
    @JsonProperty("rms_norm_eps")
    private double rmsNormEps = 1e-5;

    /** Layer norm epsilon (GPT-2, BERT-style). */
    @JsonProperty("layer_norm_eps")
    private double layerNormEps = 1e-5;

    // ── Rotary position embedding ─────────────────────────────────────────────

    /** RoPE base frequency (default 10000 for most models, 500000 for LLaMA-3). */
    @JsonProperty("rope_theta")
    private double ropeTheta = 10000.0;

    /** Per-dimension RoPE scaling factor (LLaMA-2 long-context). */
    @JsonProperty("rope_scaling")
    private RopeScaling ropeScaling;

    // ── Gemma-4 layer-specific RoPE ─────────────────────────────────────────
    @JsonProperty("layer_types")
    private List<String> layerTypes;

    private Double ropeThetaFull;
    private Double ropeThetaSliding;
    private Double partialRotaryFactorFull;
    private Double partialRotaryFactorSliding;

    private Double finalLogitSoftcapping;

    // ── Activation function ───────────────────────────────────────────────────

    /** Activation in the FFN: "silu", "gelu", "relu", "gelu_new". */
    @JsonProperty("hidden_act")
    private String hiddenAct = "silu";

    // ── Special tokens ────────────────────────────────────────────────────────

    @JsonProperty("bos_token_id")
    private Integer bosTokenId;
    @JsonProperty("eos_token_id")
    private Integer eosTokenId;
    @JsonProperty("pad_token_id")
    private Integer padTokenId;

    // ── Dtype ─────────────────────────────────────────────────────────────────

    /** Storage dtype declared in config: "bfloat16", "float16", "float32". */
    @JsonProperty("torch_dtype")
    private String torchDtype = "bfloat16";

    // ── Tie embeddings ────────────────────────────────────────────────────────

    /**
     * Whether {@code lm_head.weight} is tied (shared) with
     * {@code model.embed_tokens.weight}.
     */
    @JsonProperty("tie_word_embeddings")
    private boolean tieWordEmbeddings = false;

    // ── Sliding window attention (Mistral) ────────────────────────────────────

    @JsonProperty("sliding_window")
    private Integer slidingWindow; // null = full attention

    // ── Head dimension override (Gemma2, some Phi variants) ──────────────────

    @JsonProperty("head_dim")
    private Integer headDim; // null = hiddenSize / numAttentionHeads

    // ── MoE fields ──────────────────────────────────────────────────────────

    @JsonProperty("num_local_experts")
    private Integer numLocalExperts;

    @JsonProperty("num_experts_per_tok")
    private Integer numExpertsPerTok;

    @JsonProperty("decoder_sparse_step")
    private Integer decoderSparseStep;

    // ─────────────────────────────────────────────────────────────────────────
    // Factory / loading
    // ─────────────────────────────────────────────────────────────────────────

    /** Parse {@code config.json} from an explicit file path. */
    public static ModelConfig load(Path configPath, ObjectMapper mapper) throws IOException {
        if (!Files.exists(configPath)) {
            throw new IOException("config.json not found: " + configPath);
        }
        ModelConfig cfg = mapper.readValue(configPath.toFile(), ModelConfig.class);
        // Fallback for multi-modal configs where root fields may be lost or malformed.
        if ((cfg.modelType == null || cfg.modelType.isBlank())
                && (cfg.architectures == null || cfg.architectures.isEmpty())) {
            try {
                JsonNode root = mapper.readTree(configPath.toFile());
                String inferredType = textValue(root, "model_type");
                List<String> inferredArch = listValue(root, "architectures");

                if (inferredType == null || inferredType.isBlank()) {
                    inferredType = textValue(root.path("text_config"), "model_type");
                }
                if (inferredArch == null || inferredArch.isEmpty()) {
                    inferredArch = listValue(root.path("text_config"), "architectures");
                }
                if ((inferredType != null && !inferredType.isBlank())
                        || (inferredArch != null && !inferredArch.isEmpty())) {
                    cfg.modelType = inferredType;
                    cfg.architectures = inferredArch;
                    log.infof("ModelConfig: inferred fields from fallback: type=%s arch=%s",
                            cfg.modelType, cfg.primaryArchitecture());
                }
            } catch (Exception e) {
                log.debugf("ModelConfig: fallback inference failed: %s", e.getMessage());
            }
        }
        // Merge text_config values when root fields are missing (Gemma4-style configs).
        try {
            JsonNode root = mapper.readTree(configPath.toFile());
            JsonNode textCfg = root.path("text_config");
            if (textCfg != null && !textCfg.isMissingNode()) {
                if (!root.has("hidden_size")) {
                    Integer v = intValue(textCfg, "hidden_size");
                    if (v != null) cfg.hiddenSize = v;
                }
                if (!root.has("num_hidden_layers")) {
                    Integer v = intValue(textCfg, "num_hidden_layers");
                    if (v != null) cfg.numHiddenLayers = v;
                }
                if (!root.has("num_attention_heads")) {
                    Integer v = intValue(textCfg, "num_attention_heads");
                    if (v != null) cfg.numAttentionHeads = v;
                }
                if (!root.has("num_key_value_heads")) {
                    Integer v = intValue(textCfg, "num_key_value_heads");
                    if (v != null) cfg.numKeyValueHeads = v;
                }
                if (!root.has("intermediate_size")) {
                    Integer v = intValue(textCfg, "intermediate_size");
                    if (v != null) cfg.intermediateSize = v;
                }
                if (!root.has("vocab_size")) {
                    Integer v = intValue(textCfg, "vocab_size");
                    if (v != null) cfg.vocabSize = v;
                }
                if (!root.has("rms_norm_eps")) {
                    Double v = doubleValue(textCfg, "rms_norm_eps");
                    if (v != null) cfg.rmsNormEps = v;
                }
                if (!root.has("rope_theta")) {
                    Double v = doubleValue(textCfg, "rope_theta");
                    if (v != null) cfg.ropeTheta = v;
                }
                if (!root.has("head_dim")) {
                    Integer v = intValue(textCfg, "head_dim");
                    if (v != null) cfg.headDim = v;
                }
                if (!root.has("final_logit_softcapping")) {
                    Double v = doubleValue(textCfg, "final_logit_softcapping");
                    if (v != null) cfg.finalLogitSoftcapping = v;
                }
                if (!root.has("layer_types")) {
                    JsonNode types = textCfg.get("layer_types");
                    if (types != null && types.isArray()) {
                        List<String> values = new ArrayList<>();
                        types.forEach(n -> values.add(n.asText()));
                        cfg.layerTypes = values;
                    }
                }
                if (!root.has("rope_parameters")) {
                    JsonNode ropeParams = textCfg.get("rope_parameters");
                    if (ropeParams != null && ropeParams.isObject()) {
                        JsonNode full = ropeParams.get("full_attention");
                        if (full != null && full.isObject()) {
                            Double v = doubleValue(full, "rope_theta");
                            if (v != null) cfg.ropeThetaFull = v;
                            Double p = doubleValue(full, "partial_rotary_factor");
                            if (p != null) cfg.partialRotaryFactorFull = p;
                        }
                        JsonNode sliding = ropeParams.get("sliding_attention");
                        if (sliding != null && sliding.isObject()) {
                            Double v = doubleValue(sliding, "rope_theta");
                            if (v != null) cfg.ropeThetaSliding = v;
                            Double p = doubleValue(sliding, "partial_rotary_factor");
                            if (p != null) cfg.partialRotaryFactorSliding = p;
                        }
                    }
                }
                if (!root.has("hidden_size_per_layer_input")) {
                    Integer v = intValue(textCfg, "hidden_size_per_layer_input");
                    if (v != null) cfg.hiddenSizePerLayerInput = v;
                }
                if (!root.has("vocab_size_per_layer_input")) {
                    Integer v = intValue(textCfg, "vocab_size_per_layer_input");
                    if (v != null) cfg.vocabSizePerLayerInput = v;
                }
                if (!root.has("bos_token_id")) {
                    Integer v = intValue(textCfg, "bos_token_id");
                    if (v != null) cfg.bosTokenId = v;
                }
                if (!root.has("eos_token_id")) {
                    Integer v = intValue(textCfg, "eos_token_id");
                    if (v != null) cfg.eosTokenId = v;
                }
                if (!root.has("pad_token_id")) {
                    Integer v = intValue(textCfg, "pad_token_id");
                    if (v != null) cfg.padTokenId = v;
                }
            }
        } catch (Exception e) {
            log.debugf("ModelConfig: text_config merge failed: %s", e.getMessage());
        }
        log.infof("Loaded model config: type=%s arch=%s layers=%d hidden=%d heads=%d kvHeads=%d",
                cfg.modelType, cfg.primaryArchitecture(),
                cfg.numHiddenLayers, cfg.hiddenSize,
                cfg.numAttentionHeads, cfg.resolvedNumKvHeads());
        return cfg;
    }

    /**
     * Load {@code config.json} from the model directory.
     *
     * @param modelDir directory containing config.json and the safetensors weights
     */
    public static ModelConfig fromDirectory(Path modelDir, ObjectMapper mapper) throws IOException {
        return load(modelDir.resolve("config.json"), mapper);
    }

    /**
     * Check whether a model directory has a config.json.
     */
    public static boolean exists(Path modelDir) {
        return Files.exists(modelDir.resolve("config.json"));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Derived / computed accessors
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * Primary architecture class name (first entry in the list, or model_type
     * fallback).
     */
    public String primaryArchitecture() {
        if (architectures != null && !architectures.isEmpty())
            return architectures.get(0);
        return modelType != null ? modelType : "unknown";
    }

    /**
     * Accepts either a JSON array or single string for "architectures".
     */
    @JsonSetter("architectures")
    public void setArchitectures(JsonNode node) {
        if (node == null || node.isNull()) {
            this.architectures = null;
            return;
        }
        if (node.isArray()) {
            List<String> out = new java.util.ArrayList<>();
            for (JsonNode n : node) {
                if (n != null && !n.isNull()) {
                    out.add(n.asText());
                }
            }
            this.architectures = out;
            return;
        }
        if (node.isTextual()) {
            this.architectures = List.of(node.asText());
            return;
        }
        // Unknown shape, keep null so we can fall back to model_type.
        this.architectures = null;
    }

    /**
     * Direct setter for runtime overrides.
     */
    public void setArchitectures(List<String> architectures) {
        this.architectures = architectures;
    }

    @JsonSetter("bos_token_id")
    public void setBosTokenId(JsonNode node) {
        this.bosTokenId = firstInt(node);
    }

    @JsonSetter("eos_token_id")
    public void setEosTokenId(JsonNode node) {
        this.eosTokenId = firstInt(node);
    }

    @JsonSetter("pad_token_id")
    public void setPadTokenId(JsonNode node) {
        this.padTokenId = firstInt(node);
    }

    private static String textValue(JsonNode node, String field) {
        if (node == null || node.isMissingNode() || node.isNull()) {
            return null;
        }
        JsonNode v = node.get(field);
        if (v == null || v.isNull()) {
            return null;
        }
        return v.isTextual() ? v.asText() : v.toString();
    }

    private static List<String> listValue(JsonNode node, String field) {
        if (node == null || node.isMissingNode() || node.isNull()) {
            return null;
        }
        JsonNode v = node.get(field);
        if (v == null || v.isNull()) {
            return null;
        }
        if (v.isArray()) {
            List<String> out = new java.util.ArrayList<>();
            for (JsonNode n : v) {
                if (n != null && !n.isNull()) {
                    out.add(n.asText());
                }
            }
            return out;
        }
        if (v.isTextual()) {
            return List.of(v.asText());
        }
        return null;
    }

    private static Integer firstInt(JsonNode node) {
        if (node == null || node.isNull()) {
            return null;
        }
        if (node.isNumber()) {
            return node.asInt();
        }
        if (node.isArray() && node.size() > 0) {
            JsonNode first = node.get(0);
            return first != null && first.isNumber() ? first.asInt() : null;
        }
        return null;
    }

    private static Integer intValue(JsonNode node, String field) {
        if (node == null || node.isMissingNode() || node.isNull()) {
            return null;
        }
        JsonNode v = node.get(field);
        if (v == null || v.isNull()) {
            return null;
        }
        return v.isNumber() ? v.asInt() : null;
    }

    private static Double doubleValue(JsonNode node, String field) {
        if (node == null || node.isMissingNode() || node.isNull()) {
            return null;
        }
        JsonNode v = node.get(field);
        if (v == null || v.isNull()) {
            return null;
        }
        return v.isNumber() ? v.asDouble() : null;
    }

    /** Resolved number of KV heads — defaults to numAttentionHeads if not set. */
    public int resolvedNumKvHeads() {
        return numKeyValueHeads != null ? numKeyValueHeads : numAttentionHeads;
    }

    /** Per-head dimension: {@code hiddenSize / numAttentionHeads}. */
    public int resolvedHeadDim() {
        return headDim != null ? headDim : hiddenSize / numAttentionHeads;
    }

    /**
     * GQA group size: number of query heads sharing one KV head.
     * 1 = MHA, N = GQA/MQA.
     */
    public int kvGroupSize() {
        return numAttentionHeads / resolvedNumKvHeads();
    }

    /** Whether this model uses Grouped-Query Attention (GQA or MQA). */
    public boolean isGroupedQueryAttention() {
        return resolvedNumKvHeads() < numAttentionHeads;
    }

    public Optional<Integer> bosTokenId() {
        return Optional.ofNullable(bosTokenId);
    }

    public Optional<Integer> eosTokenId() {
        return Optional.ofNullable(eosTokenId);
    }

    public Optional<Integer> padTokenId() {
        return Optional.ofNullable(padTokenId);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Overrides (runtime reconciliation)
    // ─────────────────────────────────────────────────────────────────────────

    /** Override the number of attention heads when config.json is inconsistent with weights. */
    public void overrideNumAttentionHeads(int heads) {
        if (heads > 0) {
            this.numAttentionHeads = heads;
        }
    }

    /** Override the number of key-value heads when config.json is inconsistent with weights. */
    public void overrideNumKeyValueHeads(Integer heads) {
        if (heads != null && heads > 0) {
            this.numKeyValueHeads = heads;
        }
    }

    /** Override head dimension when config.json is inconsistent with weights. */
    public void overrideHeadDim(Integer dim) {
        if (dim != null && dim > 0) {
            this.headDim = dim;
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Plain getters (used by JSON deserialisation and compute kernels)
    // ─────────────────────────────────────────────────────────────────────────

    public String modelType() {
        return modelType;
    }

    public int hiddenSize() {
        return hiddenSize;
    }

    public int numHiddenLayers() {
        return numHiddenLayers;
    }

    public int numAttentionHeads() {
        return numAttentionHeads;
    }

    public int intermediateSize() {
        return intermediateSize;
    }

    public int vocabSize() {
        return vocabSize;
    }

    public int maxPositionEmbeddings() {
        return maxPositionEmbeddings;
    }

    public int hiddenSizePerLayerInput() {
        return hiddenSizePerLayerInput;
    }

    public int vocabSizePerLayerInput() {
        return vocabSizePerLayerInput;
    }

    public String layerType(int layerIdx) {
        if (layerTypes == null || layerIdx < 0 || layerIdx >= layerTypes.size()) {
            return null;
        }
        return layerTypes.get(layerIdx);
    }

    public double ropeThetaForLayer(int layerIdx) {
        String type = layerType(layerIdx);
        if ("full_attention".equals(type) && ropeThetaFull != null) {
            return ropeThetaFull;
        }
        if ("sliding_attention".equals(type) && ropeThetaSliding != null) {
            return ropeThetaSliding;
        }
        return ropeTheta;
    }

    public double partialRotaryFactorForLayer(int layerIdx) {
        String type = layerType(layerIdx);
        if ("full_attention".equals(type) && partialRotaryFactorFull != null) {
            return partialRotaryFactorFull;
        }
        if ("sliding_attention".equals(type) && partialRotaryFactorSliding != null) {
            return partialRotaryFactorSliding;
        }
        return 1.0;
    }

    public boolean isSlidingAttentionLayer(int layerIdx) {
        return "sliding_attention".equals(layerType(layerIdx));
    }

    public Double finalLogitSoftcapping() {
        return finalLogitSoftcapping;
    }

    public double rmsNormEps() {
        return rmsNormEps;
    }

    public double layerNormEps() {
        return layerNormEps;
    }

    public double ropeTheta() {
        return ropeTheta;
    }

    public RopeScaling ropeScaling() {
        return ropeScaling;
    }

    public String hiddenAct() {
        return hiddenAct;
    }

    public String torchDtype() {
        return torchDtype;
    }

    public boolean tieWordEmbeddings() {
        return tieWordEmbeddings;
    }

    public List<String> architectures() {
        return architectures;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Nested types
    // ─────────────────────────────────────────────────────────────────────────

    /**
     * RoPE scaling configuration (used for long-context variants).
     * <p>
     * Types: "linear", "dynamic", "yarn".
     */
    @JsonIgnoreProperties(ignoreUnknown = true)
    public static final class RopeScaling {
        @JsonProperty("type")
        public String type;
        @JsonProperty("factor")
        public double factor = 1.0;
        @JsonProperty("low_freq_factor")
        public double lowFreqFactor = 1.0;
        @JsonProperty("high_freq_factor")
        public double highFreqFactor = 4.0;
        @JsonProperty("original_max_position_embeddings")
        public int originalMaxPositionEmbeddings = 8192;
    }

    @Override
    public String toString() {
        return "ModelConfig{type=" + modelType
                + ", arch=" + primaryArchitecture()
                + ", layers=" + numHiddenLayers
                + ", hidden=" + hiddenSize
                + ", heads=" + numAttentionHeads + "/" + resolvedNumKvHeads()
                + ", vocab=" + vocabSize
                + '}';
    }

    /**
     * Check if model has sliding window attention.
     */
    public boolean hasSlidingWindow() {
        return slidingWindow != null && slidingWindow > 0;
    }

    /**
     * Get sliding window size.
     */
    public int slidingWindowSize() {
        return slidingWindow != null ? slidingWindow : Integer.MAX_VALUE;
    }

    /**
     * Check if model is MoE (Mixture of Experts).
     */
    public boolean isMoe() {
        return numLocalExperts != null && numLocalExperts > 1;
    }

    /**
     * Get number of local experts (for MoE models).
     */
    public int numLocalExperts() {
        return numLocalExperts != null ? numLocalExperts : 0;
    }

    /**
     * Get number of experts per token (for MoE models).
     */
    public int numExpertsPerTok() {
        return numExpertsPerTok != null ? numExpertsPerTok : 0;
    }

    /**
     * Check if layer is MoE layer (for interleaved architectures).
     */
    public boolean isMoeLayer(int layerIdx) {
        if (!isMoe())
            return false;
        int step = decoderSparseStep != null ? decoderSparseStep : 1;
        return (layerIdx + 1) % step == 0;
    }

    /**
     * Get number of key-value heads (GQA).
     */
    public int numKeyValueHeads() {
        return numKeyValueHeads;
    }
}
