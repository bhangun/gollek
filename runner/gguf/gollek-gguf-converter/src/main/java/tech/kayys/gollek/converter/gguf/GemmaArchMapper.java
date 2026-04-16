package tech.kayys.gollek.converter.gguf;

import java.util.ArrayList;
import java.util.List;

/**
 * Maps a HuggingFace Gemma model config to GGUF metadata and maps HF tensor
 * names to GGUF tensor names following the llama.cpp naming convention.
 *
 * <p>
 * This mapper handles all Gemma family variants:
 * <ul>
 * <li>Gemma 1 (original)</li>
 * <li>Gemma 2 (with sliding window, Q/K norms)</li>
 * <li>Gemma 4 (with per-layer input gating, projection, and post-input norm)</li>
 * </ul>
 *
 * <p>
 * Gemma architecture differences from LLaMA:
 * <ul>
 * <li>Uses GeGLU activation instead of SiLU</li>
 * <li>Different layer norm naming conventions</li>
 * <li>Gemma 2/4: Q and K normalization layers</li>
 * <li>Gemma 4: Per-layer input gating and projection</li>
 * <li>Gemma 4: Post per-layer input normalization</li>
 * </ul>
 */
public final class GemmaArchMapper {

    private GemmaArchMapper() {
    }

    // ─────────────────────────────────────────────────────────────────────
    // Metadata population
    // ─────────────────────────────────────────────────────────────────────

    /**
     * Populate GGUF metadata from a parsed HuggingFace {@code config.json}.
     */
    public static void applyConfig(GgufModel model,
            HfConfigParser.ModelConfig cfg,
            HfConfigParser.TokenizerData tok,
            String version) {

        String arch = "gemma";

        // ── General ──────────────────────────────────────────────────────
        model.addMeta("general.architecture",  GgufMetaValue.ofString(arch));
        model.addMeta("general.name",          GgufMetaValue.ofString(
                cfg.modelType() + "-" + version));
        model.addMeta("general.version",       GgufMetaValue.ofString(version));
        model.addMeta("general.alignment",     GgufMetaValue.ofUInt32(GgufModel.DEFAULT_ALIGNMENT));

        // ── Hyperparameters ───────────────────────────────────────────────
        String pfx = arch + ".";
        model.addMeta(pfx + "context_length",
                GgufMetaValue.ofUInt32(cfg.maxPositionEmbeddings()));
        model.addMeta(pfx + "embedding_length",
                GgufMetaValue.ofUInt32(cfg.hiddenSize()));
        model.addMeta(pfx + "block_count",
                GgufMetaValue.ofUInt32(cfg.numHiddenLayers()));
        model.addMeta(pfx + "feed_forward_length",
                GgufMetaValue.ofUInt32(cfg.intermediateSize()));
        model.addMeta(pfx + "attention.head_count",
                GgufMetaValue.ofUInt32(cfg.numAttentionHeads()));
        model.addMeta(pfx + "attention.head_count_kv",
                GgufMetaValue.ofUInt32(cfg.numKeyValueHeads()));
        model.addMeta(pfx + "attention.layer_norm_rms_epsilon",
                GgufMetaValue.ofFloat32(cfg.rmsNormEps()));
        model.addMeta(pfx + "rope.freq_base",
                GgufMetaValue.ofFloat32(cfg.ropeTheta()));
        model.addMeta(pfx + "vocab_size",
                GgufMetaValue.ofUInt32(cfg.vocabSize()));

        // head_dim = hidden_size / num_attention_heads
        int headDim = cfg.hiddenSize() / Math.max(1, cfg.numAttentionHeads());
        model.addMeta(pfx + "rope.dimension_count",
                GgufMetaValue.ofUInt32(headDim));

        // Gemma-specific: key and value lengths
        model.addMeta(pfx + "attention.key_length",
                GgufMetaValue.ofUInt32(headDim));
        model.addMeta(pfx + "attention.value_length",
                GgufMetaValue.ofUInt32(headDim));

        // attention bias flag
        if (cfg.attentionBias()) {
            model.addMeta(pfx + "attention.clamp_kqv",
                    GgufMetaValue.ofBool(true));
        }

        // partial rotary factor
        if (cfg.partialRotaryFactor() > 0f && cfg.partialRotaryFactor() < 1f) {
            model.addMeta(pfx + "rope.dimension_count",
                    GgufMetaValue.ofUInt32(
                            (int) (headDim * cfg.partialRotaryFactor())));
        }

        // sliding window (Gemma 2/4)
        if (cfg.slidingWindow() > 0) {
            model.addMeta(pfx + "attention.sliding_window",
                    GgufMetaValue.ofUInt32(cfg.slidingWindow()));
        }

        // RoPE scaling
        applyRopeScaling(model, pfx, cfg);

        // ── Architecture-specific metadata ────────────────────────────────
        applyArchSpecificMetadata(model, arch, cfg);

        // ── Tokenizer ─────────────────────────────────────────────────────
        if (tok != null) {
            model.addMeta("tokenizer.ggml.model",
                    GgufMetaValue.ofString(mapTokenizerModel(tok.tokenizerModel())));
            if (!tok.vocab().isEmpty()) {
                model.addMeta("tokenizer.ggml.tokens",
                        GgufMetaValue.ofStringArray(tok.vocab()));
            }
            if (!tok.scores().isEmpty()) {
                model.addMeta("tokenizer.ggml.scores",
                        GgufMetaValue.ofFloat32Array(tok.scores()));
            }
            if (!tok.tokenTypes().isEmpty()) {
                model.addMeta("tokenizer.ggml.token_type",
                        GgufMetaValue.ofInt32Array(tok.tokenTypes()));
            }
            model.addMeta("tokenizer.ggml.bos_token_id",
                    GgufMetaValue.ofUInt32(tok.bosId()));
            model.addMeta("tokenizer.ggml.eos_token_id",
                    GgufMetaValue.ofUInt32(tok.eosId()));
            model.addMeta("tokenizer.ggml.padding_token_id",
                    GgufMetaValue.ofUInt32(0));
        }
    }

    private static void applyRopeScaling(GgufModel model,
            String pfx, HfConfigParser.ModelConfig cfg) {
        if (cfg.raw() == null || !cfg.raw().has("rope_scaling")) return;
        var rs = cfg.raw().get("rope_scaling");
        if (rs.isJsonNull()) return;
        var rsObj = rs.getAsJsonObject();
        if (rsObj.has("type")) {
            model.addMeta(pfx + "rope.scaling.type",
                    GgufMetaValue.ofString(rsObj.get("type").getAsString()));
        }
        if (rsObj.has("factor")) {
            model.addMeta(pfx + "rope.scaling.factor",
                    GgufMetaValue.ofFloat32(rsObj.get("factor").getAsFloat()));
        }
        if (rsObj.has("original_max_position_embeddings")) {
            model.addMeta(pfx + "rope.scaling.original_context_length",
                    GgufMetaValue.ofUInt32(
                            rsObj.get("original_max_position_embeddings").getAsLong()));
        }
        if (rsObj.has("low_freq_factor")) {
            model.addMeta(pfx + "rope.scaling.low_freq_factor",
                    GgufMetaValue.ofFloat32(rsObj.get("low_freq_factor").getAsFloat()));
        }
        if (rsObj.has("high_freq_factor")) {
            model.addMeta(pfx + "rope.scaling.high_freq_factor",
                    GgufMetaValue.ofFloat32(rsObj.get("high_freq_factor").getAsFloat()));
        }
    }

    /**
     * Apply architecture-specific metadata extensions for Gemma.
     */
    private static void applyArchSpecificMetadata(GgufModel model, String arch,
            HfConfigParser.ModelConfig cfg) {
        if (cfg.raw() == null) return;
        var raw = cfg.raw();
        String pfx = arch + ".";

        // Gemma 2/4: num_query_groups for KV heads
        if (raw.has("num_query_groups")) {
            model.addMeta(pfx + "attention.head_count_kv",
                GgufMetaValue.ofUInt32(raw.get("num_query_groups").getAsInt()));
        }

        // Gemma 4: Check for specific features
        boolean isGemma4 = cfg.modelType().toLowerCase().contains("gemma4") ||
                          cfg.modelType().toLowerCase().contains("gemma_4");
        
        if (isGemma4) {
            // Add Gemma 4 specific metadata
            model.addMeta(pfx + "architecture.variant",
                GgufMetaValue.ofString("gemma4"));
            
            // Gemma 4 uses per-layer input gating
            model.addMeta(pfx + "per_layer_input_gate",
                GgufMetaValue.ofBool(true));
            
            // Gemma 4 uses per-layer projection
            model.addMeta(pfx + "per_layer_projection",
                GgufMetaValue.ofBool(true));
            
            // Gemma 4 uses post per-layer input normalization
            model.addMeta(pfx + "post_per_layer_input_norm",
                GgufMetaValue.ofBool(true));
        }

        // Gemma 2 specific
        if (cfg.modelType().toLowerCase().contains("gemma2") ||
            cfg.modelType().toLowerCase().contains("gemma_2")) {
            model.addMeta(pfx + "architecture.variant",
                GgufMetaValue.ofString("gemma2"));
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // Tokenizer mapping
    // ─────────────────────────────────────────────────────────────────────

    public static String mapTokenizerModel(String hfType) {
        return switch (hfType.toLowerCase()) {
            case "bpe"       -> "gpt2";
            case "unigram"   -> "llama";
            case "wordpiece" -> "bert";
            default          -> hfType.toLowerCase();
        };
    }

    // ─────────────────────────────────────────────────────────────────────
    // Tensor name mapping
    // ─────────────────────────────────────────────────────────────────────

    /**
     * Map a HuggingFace weight name to a GGUF tensor name.
     * Returns {@code null} if the tensor should be skipped.
     */
    public static String mapTensorName(String hfName, int numLayers) {
        // ── Always-skip list ──────────────────────────────────────────────
        if (hfName.endsWith(".inv_freq"))                    return null;
        if (hfName.endsWith(".attention.bias"))              return null;
        if (hfName.contains("._orig_mod"))                  return null;

        // ── Top-level (non-layer) tensors ─────────────────────────────────
        String top = mapTopLevel(hfName);
        if (top != null) return top;

        // ── Per-layer tensors ─────────────────────────────────────────────
        return mapLayerTensor(hfName, numLayers);
    }

    // -- Top-level (not inside a layer) -----------------------------------

    private static String mapTopLevel(String n) {
        return switch (n) {
            // Embeddings
            case "model.embed_tokens.weight",
                 "model.language_model.embed_tokens.weight",
                 "transformer.word_embeddings.weight",
                 "transformer.embedding.word_embeddings.weight",
                 "word_embeddings.weight",
                 "backbone.embedding.weight",
                 "backbone.embeddings.weight",
                 "tok_embeddings.weight" -> "token_embd.weight";
            
            // Per-layer embedding (Gemma 4) - SKIP: Not supported by llama.cpp, too large (4.48 GB)
            case "model.language_model.embed_tokens_per_layer.weight" -> null; // Skip

            // Norm after all layers
            case "model.norm.weight",
                 "model.language_model.norm.weight",
                 "transformer.norm_f.weight",
                 "transformer.ln_f.weight",
                 "norm.weight",
                 "backbone.norm_f.weight",
                 "transformer.final_layernorm.weight" -> "output_norm.weight";
            case "model.norm.bias",
                 "model.language_model.norm.bias",
                 "transformer.norm_f.bias",
                 "transformer.ln_f.bias",
                 "norm.bias",
                 "transformer.final_layernorm.bias" -> "output_norm.bias";

            // LM head / output projection
            case "lm_head.weight",
                 "output.weight",
                 "embed_out.weight" -> "output.weight";
            case "lm_head.bias" -> "output.bias";
            
            // Gemma 4 specific: per-layer projection - SKIP: Not standard GGUF
            case "model.language_model.per_layer_model_projection.weight" -> null; // Skip
            case "model.language_model.per_layer_projection_norm.weight" -> null; // Skip

            // Token type embeddings (BERT-style)
            case "embeddings.token_type_embeddings.weight" -> "token_types.weight";
            case "embeddings.position_embeddings.weight"   -> "pos_embd.weight";

            default -> null;
        };
    }

    // -- Per-layer --------------------------------------------------------

    private static String mapLayerTensor(String n, int numLayers) {
        // HuggingFace standard: model.layers.{i}.xxx
        // Gemma 4 multimodal: model.language_model.layers.{i}.xxx
        String[] prefixes = {
            "model.language_model.layers.", // Gemma 4 multimodal
            "model.layers.",                // LLaMA, Mistral, Qwen2, Gemma, DeepSeek
            "transformer.h.",               // GPT-2, Bloom, Falcon
            "transformer.blocks.",          // MPT
            "model.decoder.layers.",        // OPT
            "transformer.layers.",          // Baichuan, InternLM
            "layers.",                      // older LLaMA PyTorch
            "backbone.layers.",             // Mamba
        };

        for (String prefix : prefixes) {
            if (n.startsWith(prefix)) {
                String rest = n.substring(prefix.length());
                int dot = rest.indexOf('.');
                if (dot < 0) return null;
                int layerId;
                try { layerId = Integer.parseInt(rest.substring(0, dot)); }
                catch (NumberFormatException e) { continue; }
                String tail = rest.substring(dot + 1);
                String mapped = mapTail(tail, layerId);
                return mapped; // null means skip
            }
        }
        return null;
    }

    /**
     * Map the tail of a layer-tensor name (everything after
     * {@code model.layers.N.}) to a GGUF key inside {@code blk.N.}.
     */
    @SuppressWarnings("DuplicateBranchesInSwitch")
    private static String mapTail(String tail, int layerId) {
        String pfx = "blk." + layerId + ".";
        String mapped = switch (tail) {

            // ── Attention projections ─────────────────────────────────────
            case "self_attn.q_proj.weight"                  -> pfx + "attn_q.weight";
            case "self_attn.k_proj.weight"                  -> pfx + "attn_k.weight";
            case "self_attn.v_proj.weight"                  -> pfx + "attn_v.weight";
            case "self_attn.o_proj.weight"                  -> pfx + "attn_output.weight";

            // Biases on attention projections
            case "self_attn.q_proj.bias"                    -> pfx + "attn_q.bias";
            case "self_attn.k_proj.bias"                    -> pfx + "attn_k.bias";
            case "self_attn.v_proj.bias"                    -> pfx + "attn_v.bias";
            case "self_attn.o_proj.bias"                    -> pfx + "attn_output.bias";

            // ── MLP / FFN (Gemma uses GeGLU) ──────────────────────────────
            case "mlp.gate_proj.weight"                     -> pfx + "ffn_gate.weight";
            case "mlp.up_proj.weight"                       -> pfx + "ffn_up.weight";
            case "mlp.down_proj.weight"                     -> pfx + "ffn_down.weight";

            // Biases on MLP
            case "mlp.gate_proj.bias"                       -> pfx + "ffn_gate.bias";
            case "mlp.up_proj.bias"                         -> pfx + "ffn_up.bias";
            case "mlp.down_proj.bias"                       -> pfx + "ffn_down.bias";

            // ── Layer norms ───────────────────────────────────────────────
            case "input_layernorm.weight"                   -> pfx + "attn_norm.weight";
            case "input_layernorm.bias"                     -> pfx + "attn_norm.bias";

            case "post_attention_layernorm.weight"          -> pfx + "ffn_norm.weight";
            case "post_attention_layernorm.bias"            -> pfx + "ffn_norm.bias";

            // Pre-layer norm (Gemma 2/4)
            case "pre_feedforward_layernorm.weight"         -> pfx + "ffn_pre_norm.weight";
            case "post_feedforward_layernorm.weight"        -> pfx + "ffn_post_norm.weight";

            // ── Gemma 2/4: Q and K normalization ──────────────────────────
            case "self_attn.q_norm.weight"                  -> pfx + "attn_q_norm.weight";
            case "self_attn.k_norm.weight"                  -> pfx + "attn_k_norm.weight";

            // ── Gemma 4: Per-layer input gating - SKIP: Not supported by llama.cpp
            case "per_layer_input_gate.weight"              -> null; // Skip
            case "per_layer_projection.weight"              -> null; // Skip
            case "post_per_layer_input_norm.weight"         -> null; // Skip
            case "layer_scalar"                             -> null; // Skip

            // ── Embedding norm (some models add it per layer) ─────────────
            case "norm.weight"                              -> pfx + "layer_out_norm.weight";
            case "norm.bias"                                -> pfx + "layer_out_norm.bias";

            // ── Rotary embedding (skip) ───────────────────────────────────
            case "rotary_emb.inv_freq"                      -> null;

            default                                         -> null; // skip unknown
        };

        return mapped;
    }
}
