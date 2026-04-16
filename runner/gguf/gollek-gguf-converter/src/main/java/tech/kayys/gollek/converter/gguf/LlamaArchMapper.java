package tech.kayys.gollek.converter.gguf;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Maps a HuggingFace model config to GGUF metadata and maps HF tensor
 * names to GGUF tensor names following the llama.cpp naming convention.
 *
 * <p>
 * Tensor name mappings are sourced from
 * {@code gguf-py/gguf/tensor_mapping.py} in llama.cpp and cover:
 * <ul>
 * <li>LLaMA / Mistral / Mixtral</li>
 * <li>Qwen2 / Qwen2-MoE</li>
 * <li>Phi-2 / Phi-3</li>
 * <li>Gemma / Gemma2</li>
 * <li>Falcon</li>
 * <li>GPT-NeoX / StableLM</li>
 * <li>Bloom / BigCode / StarCoder2</li>
 * <li>Baichuan / InternLM2</li>
 * <li>DeepSeek / DeepSeek-V2</li>
 * <li>Command-R (Cohere)</li>
 * <li>Jamba / Mamba</li>
 * <li>MPT</li>
 * </ul>
 */
public final class LlamaArchMapper {

    private LlamaArchMapper() {
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

        String arch = mapArch(cfg.modelType());

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

        // attention bias flag
        if (cfg.attentionBias()) {
            model.addMeta(pfx + "attention.clamp_kqv",
                    GgufMetaValue.ofBool(true));
        }

        // partial rotary factor (e.g. Phi models use 0.4)
        if (cfg.partialRotaryFactor() > 0f && cfg.partialRotaryFactor() < 1f) {
            model.addMeta(pfx + "rope.dimension_count",
                    GgufMetaValue.ofUInt32(
                            (int) (headDim * cfg.partialRotaryFactor())));
        }

        // sliding window
        if (cfg.slidingWindow() > 0) {
            model.addMeta(pfx + "attention.sliding_window",
                    GgufMetaValue.ofUInt32(cfg.slidingWindow()));
        }

        // RoPE scaling
        applyRopeScaling(model, pfx, cfg);

        // ── Architecture-specific extensions ─────────────────────────────
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
     * Apply architecture-specific metadata extensions.
     */
    private static void applyArchSpecificMetadata(GgufModel model, String arch,
            HfConfigParser.ModelConfig cfg) {
        if (cfg.raw() == null) return;
        var raw = cfg.raw();
        String pfx = arch + ".";

        // ── DeepSeek V2/V3 MLA (Multi-head Latent Attention) ─────────────
        if (arch.equals("deepseek2") || cfg.modelType().toLowerCase().contains("deepseek")) {
            if (raw.has("kv_lora_rank")) {
                model.addMeta(pfx + "attention.kv_lora_rank",
                    GgufMetaValue.ofUInt32(raw.get("kv_lora_rank").getAsLong()));
            }
            if (raw.has("q_lora_rank")) {
                model.addMeta(pfx + "attention.q_lora_rank",
                    GgufMetaValue.ofUInt32(raw.get("q_lora_rank").getAsLong()));
            }
            if (raw.has("qk_rope_head_dim")) {
                model.addMeta(pfx + "rope.dimension_count",
                    GgufMetaValue.ofUInt32(raw.get("qk_rope_head_dim").getAsLong()));
            }
            if (raw.has("v_head_dim")) {
                model.addMeta(pfx + "attention.value_length",
                    GgufMetaValue.ofUInt32(raw.get("v_head_dim").getAsLong()));
            }
            // MoE configuration
            if (raw.has("num_experts")) {
                model.addMeta(pfx + "moe.expert_count",
                    GgufMetaValue.ofUInt32(raw.get("num_experts").getAsLong()));
            }
            if (raw.has("num_experts_per_tok")) {
                model.addMeta(pfx + "moe.experts_per_tok",
                    GgufMetaValue.ofUInt32(raw.get("num_experts_per_tok").getAsLong()));
            }
            if (raw.has("n_routed_experts")) {
                model.addMeta(pfx + "moe.routed_expert_count",
                    GgufMetaValue.ofUInt32(raw.get("n_routed_experts").getAsLong()));
            }
            if (raw.has("n_shared_experts")) {
                model.addMeta(pfx + "moe.shared_expert_count",
                    GgufMetaValue.ofUInt32(raw.get("n_shared_experts").getAsLong()));
            }
        }

        // ── Gemma / Gemma2 specific ───────────────────────────────────────
        if (arch.equals("gemma") || arch.equals("gemma2")) {
            int headDim = cfg.hiddenSize() / Math.max(1, cfg.numAttentionHeads());
            model.addMeta(pfx + "attention.key_length",
                GgufMetaValue.ofUInt32(headDim));
            model.addMeta(pfx + "attention.value_length",
                GgufMetaValue.ofUInt32(headDim));
            if (arch.equals("gemma2")) {
                if (raw.has("sliding_window")) {
                    model.addMeta(pfx + "attention.sliding_window",
                        GgufMetaValue.ofUInt32(raw.get("sliding_window").getAsInt()));
                }
                if (raw.has("num_query_groups")) {
                    model.addMeta(pfx + "attention.head_count_kv",
                        GgufMetaValue.ofUInt32(raw.get("num_query_groups").getAsInt()));
                }
            }
        }

        // ── Qwen2-MoE specific ────────────────────────────────────────────
        if (arch.equals("qwen2") && raw.has("num_experts")) {
            if (raw.has("num_experts")) {
                model.addMeta(pfx + "moe.expert_count",
                    GgufMetaValue.ofUInt32(raw.get("num_experts").getAsLong()));
            }
            if (raw.has("num_experts_per_tok")) {
                model.addMeta(pfx + "moe.experts_per_tok",
                    GgufMetaValue.ofUInt32(raw.get("num_experts_per_tok").getAsLong()));
            }
            if (raw.has("n_shared_experts")) {
                model.addMeta(pfx + "moe.shared_expert_count",
                    GgufMetaValue.ofUInt32(raw.get("n_shared_experts").getAsLong()));
            }
        }

        // ── Phi-3 specific ────────────────────────────────────────────────
        if (arch.equals("phi3")) {
            if (raw.has("sliding_window")) {
                model.addMeta(pfx + "attention.sliding_window",
                    GgufMetaValue.ofUInt32(raw.get("sliding_window").getAsInt()));
            }
            if (raw.has("attention_multiplier")) {
                model.addMeta(pfx + "attention.scale",
                    GgufMetaValue.ofFloat32(raw.get("attention_multiplier").getAsFloat()));
            }
        }

        // ── Mamba specific ────────────────────────────────────────────────
        if (arch.equals("mamba")) {
            if (raw.has("d_conv")) {
                model.addMeta(pfx + "ssm.conv_kernel",
                    GgufMetaValue.ofUInt32(raw.get("d_conv").getAsLong()));
            }
            if (raw.has("d_inner")) {
                model.addMeta(pfx + "ssm.inner_size",
                    GgufMetaValue.ofUInt32(raw.get("d_inner").getAsLong()));
            }
            if (raw.has("d_state")) {
                model.addMeta(pfx + "ssm.state_size",
                    GgufMetaValue.ofUInt32(raw.get("d_state").getAsLong()));
            }
        }

        // ── Yi specific ───────────────────────────────────────────────────
        if (arch.equals("yi")) {
            int headDim = cfg.hiddenSize() / Math.max(1, cfg.numAttentionHeads());
            model.addMeta(pfx + "attention.scale",
                GgufMetaValue.ofFloat32((float)(1.0 / Math.sqrt(headDim))));
            if (raw.has("rope_scaling")) {
                var rs = raw.getAsJsonObject("rope_scaling");
                if (rs.has("type")) {
                    model.addMeta(pfx + "rope.scaling.type",
                        GgufMetaValue.ofString(rs.get("type").getAsString()));
                }
                if (rs.has("factor")) {
                    model.addMeta(pfx + "rope.scaling.factor",
                        GgufMetaValue.ofFloat32(rs.get("factor").getAsFloat()));
                }
            }
        }

        // ── DBRX specific (MoE) ───────────────────────────────────────────
        if (arch.equals("dbrx")) {
            if (raw.has("num_experts")) {
                model.addMeta(pfx + "moe.expert_count",
                    GgufMetaValue.ofUInt32(raw.get("num_experts").getAsLong()));
            }
            if (raw.has("num_experts_per_tok")) {
                model.addMeta(pfx + "moe.experts_per_tok",
                    GgufMetaValue.ofUInt32(raw.get("num_experts_per_tok").getAsLong()));
            }
            model.addMeta(pfx + "moe.router_bias",
                GgufMetaValue.ofFloat32(0.0f));
        }

        // ── Grok specific ─────────────────────────────────────────────────
        if (arch.equals("grok")) {
            model.addMeta(pfx + "attention.head_count_kv",
                GgufMetaValue.ofUInt32(8)); // Grok uses 8 KV heads
            model.addMeta(pfx + "attention.kq_scale",
                GgufMetaValue.ofFloat32((float)(1.0 / Math.sqrt(64))));
            model.addMeta(pfx + "rope.scaling.type",
                GgufMetaValue.ofString("yarn"));
            if (raw.has("rope_scaling")) {
                var rs = raw.getAsJsonObject("rope_scaling");
                if (rs.has("factor")) {
                    model.addMeta(pfx + "rope.scaling.factor",
                        GgufMetaValue.ofFloat32(rs.get("factor").getAsFloat()));
                }
            }
        }

        // ── InternLM2 specific ────────────────────────────────────────────
        if (arch.equals("internlm2")) {
            model.addMeta(pfx + "attention.softmax_scale",
                GgufMetaValue.ofFloat32(1.0f));
            if (raw.has("rope_scaling")) {
                var rs = raw.getAsJsonObject("rope_scaling");
                if (rs.has("type")) {
                    model.addMeta(pfx + "rope.scaling.type",
                        GgufMetaValue.ofString(rs.get("type").getAsString()));
                }
                if (rs.has("factor")) {
                    model.addMeta(pfx + "rope.scaling.factor",
                        GgufMetaValue.ofFloat32(rs.get("factor").getAsFloat()));
                }
            }
        }

        // ── Baichuan specific ─────────────────────────────────────────────
        if (arch.equals("baichuan")) {
            model.addMeta(pfx + "attention.bias",
                GgufMetaValue.ofBool(true));
            int headDim = cfg.hiddenSize() / Math.max(1, cfg.numAttentionHeads());
            model.addMeta(pfx + "attention.scale",
                GgufMetaValue.ofFloat32((float)(1.0f / Math.sqrt(headDim))));
            // Baichuan uses ALiBi positional embeddings
            model.addMeta(pfx + "positional_embedding.type",
                GgufMetaValue.ofString("alibi"));
        }

        // ── Falcon specific ───────────────────────────────────────────────
        if (arch.equals("falcon")) {
            model.addMeta(pfx + "attention.bias",
                GgufMetaValue.ofBool(true));
            model.addMeta(pfx + "attention.use_alibi",
                GgufMetaValue.ofBool(true));
            // Falcon uses parallel attention/MLP
            model.addMeta(pfx + "parallel_residual",
                GgufMetaValue.ofBool(true));
        }

        // ── MPT specific ──────────────────────────────────────────────────
        if (arch.equals("mpt")) {
            if (raw.has("clip_qkv")) {
                model.addMeta(pfx + "attention.clip_qkv",
                    GgufMetaValue.ofFloat32(raw.get("clip_qkv").getAsFloat()));
            }
            model.addMeta(pfx + "attention.alibi_slopes",
                GgufMetaValue.ofBool(true));
            model.addMeta(pfx + "positional_embedding.type",
                GgufMetaValue.ofString("alibi"));
        }

        // ── RWKV specific ─────────────────────────────────────────────────
        if (arch.equals("rwkv")) {
            model.addMeta(pfx + "time_mix_extra_dim",
                GgufMetaValue.ofUInt32(64));
            model.addMeta(pfx + "channel_mix_extra_dim",
                GgufMetaValue.ofUInt32(64));
        }

        // ── OLMo specific ─────────────────────────────────────────────────
        if (arch.equals("olmo")) {
            model.addMeta(pfx + "attention.clip_qkv",
                GgufMetaValue.ofFloat32(0.0f));
            model.addMeta(pfx + "attention.bias",
                GgufMetaValue.ofBool(true));
            model.addMeta(pfx + "mlp.bias",
                GgufMetaValue.ofBool(true));
            model.addMeta(pfx + "norm.bias",
                GgufMetaValue.ofBool(false));
        }

        // ── Command-R specific ────────────────────────────────────────────
        if (arch.equals("command-r")) {
            model.addMeta(pfx + "attention.clamp_kqv",
                GgufMetaValue.ofBool(true));
            model.addMeta(pfx + "attention.kqv_clamp_value",
                GgufMetaValue.ofFloat32(30.0f));
            model.addMeta(pfx + "logit_scale",
                GgufMetaValue.ofFloat32(0.0625f));
        }

        // ── Jamba specific (Hybrid Mamba + Attention) ─────────────────────
        if (arch.equals("jamba")) {
            model.addMeta(pfx + "hybrid_layers",
                GgufMetaValue.ofString("mamba,attention"));
            if (raw.has("layer_types")) {
                var types = raw.getAsJsonArray("layer_types");
                List<Integer> attentionLayers = new ArrayList<>();
                List<Integer> mambaLayers = new ArrayList<>();
                for (int i = 0; i < types.size(); i++) {
                    if (types.get(i).getAsString().equals("attention")) {
                        attentionLayers.add(i);
                    } else if (types.get(i).getAsString().equals("mamba")) {
                        mambaLayers.add(i);
                    }
                }
                if (!attentionLayers.isEmpty()) {
                    model.addMeta(pfx + "attention_layers",
                        GgufMetaValue.ofInt32Array(attentionLayers));
                }
                if (!mambaLayers.isEmpty()) {
                    model.addMeta(pfx + "mamba_layers",
                        GgufMetaValue.ofInt32Array(mambaLayers));
                }
            }
        }

        // ── Arctic specific ───────────────────────────────────────────────
        if (arch.equals("arctic")) {
            if (raw.has("num_experts")) {
                model.addMeta(pfx + "moe.expert_count",
                    GgufMetaValue.ofUInt32(raw.get("num_experts").getAsLong()));
            }
            if (raw.has("num_experts_per_tok")) {
                model.addMeta(pfx + "moe.experts_per_tok",
                    GgufMetaValue.ofUInt32(raw.get("num_experts_per_tok").getAsLong()));
            }
        }

        // ── BERT specific (for embeddings) ────────────────────────────────
        if (arch.equals("bert")) {
            model.addMeta(pfx + "pooling_type",
                GgufMetaValue.ofString("cls"));
            if (raw.has("max_position_embeddings")) {
                model.addMeta(pfx + "context_length",
                    GgufMetaValue.ofUInt32(raw.get("max_position_embeddings").getAsLong()));
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // Architecture name mapping
    // ─────────────────────────────────────────────────────────────────────

    public static String mapArch(String modelType) {
        String lower = modelType.toLowerCase();
        return switch (lower) {
            // LLaMA family
            case "llama", "mistral", "mistral_next", "mixtral" -> "llama";
            case "codellama", "codellama_7b", "codellama_13b", "codellama_34b" -> "llama";
            case "nous", "nous_hermes", "nous_capybara" -> "llama";
            
            // Qwen family
            case "qwen2", "qwen2_moe", "qwen2_vl", "qwen2_5" -> "qwen2";
            case "qwen", "qwen_7b", "qwen_14b", "qwen_72b" -> "qwen";
            
            // Phi family
            case "phi", "phi-msft", "phi2", "phi_2" -> "phi2";
            case "phi3", "phi3_small", "phi3_medium", "phi3_mini" -> "phi3";
            
            // Gemma family
            case "gemma", "gemma2", "gemma_2", "gemma2_2b", "gemma2_9b", "gemma2_27b" -> "gemma";
            
            // Falcon family
            case "falcon", "falcon_7b", "falcon_40b", "falcon_180b", "falcon_mamba" -> "falcon";
            
            // BLOOM family
            case "bloom", "bloom_560m", "bloom_1b7", "bloom_3b", "bloom_7b1" -> "bloom";
            
            // GPT-2 family
            case "gpt2", "gpt2_medium", "gpt2_large", "gpt2_xl" -> "gpt2";
            
            // GPT-NeoX family
            case "gpt_neox", "gpt_neox_20b", "stablelm", "stablelm_epoch", "stablelm_2" -> "gptneox";
            
            // MPT family
            case "mpt", "mpt_7b", "mpt_30b", "mpt_1b", "mpt_3b" -> "mpt";
            
            // StarCoder family
            case "starcoder2", "starcoder2_3b", "starcoder2_7b", "starcoder2_15b", "bigcode" -> "starcoder2";
            
            // Cohere family
            case "cohere", "command-r", "command-r-plus", "command-r7b", "command-r-v01" -> "command-r";
            
            // InternLM family
            case "internlm2", "internlm2_1_8b", "internlm2_7b", "internlm2_20b" -> "internlm2";
            
            // Baichuan family
            case "baichuan", "baichuan_7b", "baichuan_13b", "baichuan2_7b", "baichuan2_13b" -> "baichuan";
            
            // DeepSeek family
            case "deepseek", "deepseek_v2", "deepseek_v3", "deepseek_coder", "deepseek_r1" -> "deepseek2";
            
            // Jamba family
            case "jamba", "jamba_1_5_mini", "jamba_1_5_large", "jamba_hybrid" -> "jamba";
            
            // Mamba family
            case "mamba", "mamba_130m", "mamba_370m", "mamba_790m", "mamba_1_4b", "mamba_2_8b" -> "mamba";
            
            // RWKV family
            case "rwkv", "rwkv_4", "rwkv_5", "rwkv_6", "rwkv_world" -> "rwkv";
            
            // OLMo family
            case "olmo", "olmo_1b", "olmo_7b", "olmo_2_7b", "olmo_2_13b", "olmoe" -> "olmo";
            
            // Orion family
            case "orion", "orion_14b", "orion_7b" -> "orion";
            
            // Arctic family
            case "arctic", "arctic_7b", "arctic_480b" -> "arctic";
            
            // Xverse family
            case "xverse", "xverse_7b", "xverse_13b", "xverse_65b" -> "xverse";
            
            // ChatGLM family
            case "chatglm", "chatglm2", "chatglm3", "glm4", "glm4_9b" -> "chatglm";
            
            // Solar family
            case "solar", "solar_10_7b" -> "solar";
            
            // Granite family
            case "granite", "granite_3b", "granite_8b", "granite_20b", "granite_34b" -> "granite";
            
            // Yi family
            case "yi", "yi_6b", "yi_34b", "yi_9b" -> "yi";
            
            // DBRX family
            case "dbrx", "dbrx_132b" -> "dbrx";
            
            // Grok family
            case "grok", "grok_1" -> "grok";
            
            // BERT family (for embeddings)
            case "bert", "bert_base", "bert_large", "roberta", "distilbert" -> "bert";
            
            // T5 family (for encoder-decoder)
            case "t5", "t5_small", "t5_base", "t5_large", "t5_3b", "t5_11b" -> "t5";
            
            // Whisper (audio)
            case "whisper", "whisper_tiny", "whisper_base", "whisper_small", "whisper_medium", "whisper_large" -> "whisper";
            
            // CLIP (vision)
            case "clip", "clip_vit", "clip_vit_large", "clip_vit_huge" -> "clip";
            
            default -> lower;
        };
    }

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
     *
     * <p>Mappings mirror {@code gguf-py/gguf/tensor_mapping.py} from
     * llama.cpp (commit f486ce9f, 2026-02).
     */
    public static String mapTensorName(String hfName, int numLayers) {
        // ── Always-skip list ──────────────────────────────────────────────
        if (hfName.endsWith(".inv_freq"))                    return null;
        if (hfName.endsWith(".attention.masked_bias"))       return null;
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
                 "transformer.word_embeddings.weight",
                 "transformer.embedding.word_embeddings.weight",
                 "word_embeddings.weight",
                 "backbone.embedding.weight",
                 "backbone.embeddings.weight",
                 "tok_embeddings.weight" -> "token_embd.weight";

            // Norm after all layers
            case "model.norm.weight",
                 "transformer.norm_f.weight",
                 "transformer.ln_f.weight",
                 "norm.weight",
                 "backbone.norm_f.weight",
                 "transformer.final_layernorm.weight" -> "output_norm.weight";
            case "model.norm.bias",
                 "transformer.norm_f.bias",
                 "transformer.ln_f.bias",
                 "norm.bias",
                 "transformer.final_layernorm.bias" -> "output_norm.bias";

            // LM head / output projection
            case "lm_head.weight",
                 "output.weight",
                 "embed_out.weight" -> "output.weight";
            case "lm_head.bias" -> "output.bias";

            // Token type embeddings (BERT-style)
            case "embeddings.token_type_embeddings.weight" -> "token_types.weight";
            case "embeddings.position_embeddings.weight"   -> "pos_embd.weight";

            default -> null;
        };
    }

    // -- Per-layer --------------------------------------------------------

    private static String mapLayerTensor(String n, int numLayers) {
        // HuggingFace standard: model.layers.{i}.xxx
        String[] prefixes = {
            "model.layers.",           // LLaMA, Mistral, Qwen2, Gemma, DeepSeek
            "transformer.h.",          // GPT-2, Bloom, Falcon
            "transformer.blocks.",     // MPT
            "model.decoder.layers.",   // OPT
            "transformer.layers.",     // Baichuan, InternLM
            "layers.",                 // older LLaMA PyTorch
            "backbone.layers.",        // Mamba
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
            // LLaMA / Mistral / Qwen2 / DeepSeek / Gemma / Phi-3
            case "self_attn.q_proj.weight",
                 "attention.q_proj.weight"                  -> pfx + "attn_q.weight";
            case "self_attn.k_proj.weight",
                 "attention.k_proj.weight"                  -> pfx + "attn_k.weight";
            case "self_attn.v_proj.weight",
                 "attention.v_proj.weight"                  -> pfx + "attn_v.weight";
            case "self_attn.o_proj.weight",
                 "attention.out_proj.weight",               // GPT-2 / Bloom
                 "self_attention.dense.weight",             // Falcon
                 "attn.out_proj.weight",                    // MPT
                 "attn.c_proj.weight"                       -> pfx + "attn_output.weight";

            // Biases on attention projections
            case "self_attn.q_proj.bias",
                 "attention.q_proj.bias"                    -> pfx + "attn_q.bias";
            case "self_attn.k_proj.bias",
                 "attention.k_proj.bias"                    -> pfx + "attn_k.bias";
            case "self_attn.v_proj.bias",
                 "attention.v_proj.bias"                    -> pfx + "attn_v.bias";
            case "self_attn.o_proj.bias",
                 "attention.out_proj.bias",
                 "self_attention.dense.bias"                -> pfx + "attn_output.bias";

            // Combined QKV (Falcon, some Phi, BLOOM)
            case "self_attn.qkv_proj.weight",
                 "self_attention.query_key_value.weight",
                 "attn.Wqkv.weight"                         -> pfx + "attn_qkv.weight";
            case "self_attn.qkv_proj.bias",
                 "self_attention.query_key_value.bias",
                 "attn.Wqkv.bias"                           -> pfx + "attn_qkv.bias";

            // GPT-2 style c_attn
            case "attn.c_attn.weight"                       -> pfx + "attn_qkv.weight";
            case "attn.c_attn.bias"                         -> pfx + "attn_qkv.bias";

            // ── MLP / FFN ─────────────────────────────────────────────────
            // LLaMA-style SwiGLU (gate + up)
            case "mlp.gate_proj.weight",
                 "mlp.w1.weight"                            -> pfx + "ffn_gate.weight";
            case "mlp.up_proj.weight",
                 "mlp.w3.weight"                            -> pfx + "ffn_up.weight";
            case "mlp.down_proj.weight",
                 "mlp.w2.weight"                            -> pfx + "ffn_down.weight";

            // Biases on MLP
            case "mlp.gate_proj.bias"                       -> pfx + "ffn_gate.bias";
            case "mlp.up_proj.bias"                         -> pfx + "ffn_up.bias";
            case "mlp.down_proj.bias"                       -> pfx + "ffn_down.bias";

            // GPT-2 / Bloom / Falcon 2-layer MLP
            case "mlp.fc_in.weight",
                 "mlp.fc1.weight",
                 "mlp.dense_h_to_4h.weight",               // Bloom
                 "mlp.dense_h_to_4h_list.0.weight",        // Falcon-180B
                 "attn.c_fc.weight"                         // GPT-2
                 -> pfx + "ffn_up.weight";
            case "mlp.fc_out.weight",
                 "mlp.fc2.weight",
                 "mlp.dense_4h_to_h.weight"                 // Bloom
                 -> pfx + "ffn_down.weight";

            // Biases for 2-layer MLP
            case "mlp.fc_in.bias",  "mlp.fc1.bias",
                 "mlp.dense_h_to_4h.bias"                  -> pfx + "ffn_up.bias";
            case "mlp.fc_out.bias", "mlp.fc2.bias",
                 "mlp.dense_4h_to_h.bias"                  -> pfx + "ffn_down.bias";

            // MPT
            case "ffn.down_proj.weight"                     -> pfx + "ffn_down.weight";
            case "ffn.up_proj.weight"                       -> pfx + "ffn_up.weight";

            // ── Layer norms ───────────────────────────────────────────────
            case "input_layernorm.weight",
                 "ln_1.weight",                             // GPT-2
                 "ln_attn.weight",                          // Falcon
                 "norm_1.weight"                            -> pfx + "attn_norm.weight";
            case "input_layernorm.bias",
                 "ln_1.bias",
                 "ln_attn.bias"                             -> pfx + "attn_norm.bias";

            case "post_attention_layernorm.weight",
                 "ln_2.weight",                             // GPT-2
                 "ln_mlp.weight",                           // Falcon
                 "norm_2.weight"                            -> pfx + "ffn_norm.weight";
            case "post_attention_layernorm.bias",
                 "ln_2.bias",
                 "ln_mlp.bias"                              -> pfx + "ffn_norm.bias";

            // Pre-layer norm (some models have an extra norm before attn)
            case "pre_feedforward_layernorm.weight",
                 "pre_ffn_layernorm.weight"                 -> pfx + "ffn_pre_norm.weight";
            case "post_feedforward_layernorm.weight",
                 "post_ffn_layernorm.weight"                -> pfx + "ffn_post_norm.weight";
            case "post_attention_layernorm_pre.weight"      -> pfx + "attn_post_norm.weight";

            // Gemma2 / Phi-3 extra norms
            case "self_attn.inner_attn_ln.weight"           -> pfx + "attn_sub_norm.weight";

            // ── Embedding norm (some models add it per layer) ─────────────
            case "norm.weight"                              -> pfx + "layer_out_norm.weight";
            case "norm.bias"                                -> pfx + "layer_out_norm.bias";

            // ── Mamba / SSM layers ─────────────────────────────────────────
            case "mixer.in_proj.weight"                     -> pfx + "ssm_in.weight";
            case "mixer.out_proj.weight"                    -> pfx + "ssm_out.weight";
            case "mixer.conv1d.weight"                      -> pfx + "ssm_conv1d.weight";
            case "mixer.conv1d.bias"                        -> pfx + "ssm_conv1d.bias";
            case "mixer.x_proj.weight"                      -> pfx + "ssm_x.weight";
            case "mixer.dt_proj.weight"                     -> pfx + "ssm_dt.weight";
            case "mixer.dt_proj.bias"                       -> pfx + "ssm_dt.bias";
            case "mixer.A_log"                              -> pfx + "ssm_a";
            case "mixer.D"                                  -> pfx + "ssm_d";
            case "mixer.norm.weight"                        -> pfx + "ssm_norm.weight";

            // ── Architecture-specific variants ────────────────────────────
            // DBRX MoE tensors
            case "ffn.mlp.router.layer.weight"              -> pfx + "ffn_gate_inp.weight";
            case "ffn.mlp.experts.mlp.w1w2"                 -> pfx + "ffn_gate_up.weight";
            case "ffn.mlp.experts.mlp.w3"                   -> pfx + "ffn_down.weight";

            // Grok tensors
            case "attention.wq.weight"                      -> pfx + "attn_q.weight";
            case "attention.wk.weight"                      -> pfx + "attn_k.weight";
            case "attention.wv.weight"                      -> pfx + "attn_v.weight";
            case "attention.wo.weight"                      -> pfx + "attn_output.weight";
            case "feed_forward.w1.weight"                   -> pfx + "ffn_gate.weight";
            case "feed_forward.w2.weight"                   -> pfx + "ffn_up.weight";
            case "feed_forward.w3.weight"                   -> pfx + "ffn_down.weight";

            // RWKV specific
            case "time_mix.weight"                          -> pfx + "time_mix.weight";
            case "time_mix.key"                             -> pfx + "time_mix.key";
            case "time_mix.value"                           -> pfx + "time_mix.value";
            case "time_mix.receptance"                      -> pfx + "time_mix.receptance";
            case "time_mix.gate"                            -> pfx + "time_mix.gate";
            case "channel_mix.weight"                       -> pfx + "channel_mix.weight";

            // OLMo specific
            case "attn_proj.weight"                         -> pfx + "attn_output.weight";
            case "ff_proj.weight"                           -> pfx + "ffn_gate_up.weight";
            case "ff_out.weight"                            -> pfx + "ffn_down.weight";

            // Command-R specific (skip rotary embeddings)
            case "self_attn.rotary_emb.inv_freq"            -> null;

            // BERT-style models
            case "attention.self.query.weight"              -> pfx + "attn_q.weight";
            case "attention.self.key.weight"                -> pfx + "attn_k.weight";
            case "attention.self.value.weight"              -> pfx + "attn_v.weight";
            case "attention.output.dense.weight"            -> pfx + "attn_output.weight";
            case "intermediate.dense.weight"                -> pfx + "ffn_up.weight";
            case "output.dense.weight"                      -> pfx + "ffn_down.weight";
            case "attention.output.LayerNorm.weight"        -> pfx + "attn_norm.weight";
            case "output.LayerNorm.weight"                  -> pfx + "ffn_norm.weight";

            // ── MoE (Mixtral / DeepSeek / Qwen2-MoE) ─────────────────────
            case "block_sparse_moe.gate.weight",
                 "mlp.gate.weight"                          -> pfx + "ffn_gate_inp.weight";

            // ── Rotary embedding (some models store it explicitly) ─────────
            case "rotary_emb.inv_freq"                      -> null; // skip

            // ── StarCoder2 / Phi-2 extra ──────────────────────────────────
            case "self_attn.query.weight"                   -> pfx + "attn_q.weight";
            case "self_attn.key.weight"                     -> pfx + "attn_k.weight";
            case "self_attn.value.weight"                   -> pfx + "attn_v.weight";

            default                                         -> null; // skip unknown
        };

        // Handle MoE expert sub-layer names dynamically
        if (mapped == null) mapped = mapMoeExpert(tail, layerId);
        return mapped;
    }

    /**
     * Handle Mixture-of-Experts per-expert tensors.
     * Patterns like {@code block_sparse_moe.experts.N.w1.weight} or
     * {@code mlp.experts.N.gate_proj.weight}.
     */
    private static String mapMoeExpert(String tail, int layerId) {
        // Mixtral: block_sparse_moe.experts.{e}.{w1|w2|w3}.weight
        if (tail.startsWith("block_sparse_moe.experts.")) {
            String rest = tail.substring("block_sparse_moe.experts.".length());
            int dot = rest.indexOf('.');
            if (dot < 0) return null;
            int expertId;
            try { expertId = Integer.parseInt(rest.substring(0, dot)); }
            catch (NumberFormatException e) { return null; }
            String wtail = rest.substring(dot + 1);
            String wname = switch (wtail) {
                case "w1.weight"          -> "ffn_gate";
                case "w2.weight"          -> "ffn_down";
                case "w3.weight"          -> "ffn_up";
                case "gate_proj.weight"   -> "ffn_gate";
                case "down_proj.weight"   -> "ffn_down";
                case "up_proj.weight"     -> "ffn_up";
                default                   -> null;
            };
            if (wname == null) return null;
            return "blk." + layerId + "." + wname + ".weight." + expertId;
        }

        // DeepSeek / Qwen2-MoE: mlp.experts.{e}.gate_proj.weight
        if (tail.startsWith("mlp.experts.")) {
            String rest = tail.substring("mlp.experts.".length());
            int dot = rest.indexOf('.');
            if (dot < 0) return null;
            int expertId;
            try { expertId = Integer.parseInt(rest.substring(0, dot)); }
            catch (NumberFormatException e) { return null; }
            String wtail = rest.substring(dot + 1);
            String wname = switch (wtail) {
                case "gate_proj.weight" -> "ffn_gate";
                case "down_proj.weight" -> "ffn_down";
                case "up_proj.weight"   -> "ffn_up";
                case "w1.weight"        -> "ffn_gate";
                case "w2.weight"        -> "ffn_down";
                case "w3.weight"        -> "ffn_up";
                default                 -> null;
            };
            if (wname == null) return null;
            return "blk." + layerId + "." + wname + ".weight." + expertId;
        }

        // DeepSeek shared experts
        if (tail.startsWith("mlp.shared_experts.")) {
            String wtail = tail.substring("mlp.shared_experts.".length());
            return switch (wtail) {
                case "gate_proj.weight" -> "blk." + layerId + ".ffn_gate_shexp.weight";
                case "down_proj.weight" -> "blk." + layerId + ".ffn_down_shexp.weight";
                case "up_proj.weight"   -> "blk." + layerId + ".ffn_up_shexp.weight";
                default -> null;
            };
        }

        return null;
    }
}
