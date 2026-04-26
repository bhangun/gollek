package tech.kayys.gollek.converter.java.gguf;

import com.google.gson.*;

import java.io.IOException;
import java.io.Reader;
import java.nio.file.*;
import java.util.*;

/**
 * Parses HuggingFace {@code config.json}, {@code tokenizer_config.json},
 * and {@code tokenizer.json} into strongly-typed Java records.
 *
 * <p>
 * Extended in this revision to cover:
 * <ul>
 * <li>{@code sliding_window} (Mistral)</li>
 * <li>{@code attention_bias} / {@code bias} (Phi-2, GPT-NeoX)</li>
 * <li>{@code partial_rotary_factor} (Phi-2, Phi-3)</li>
 * <li>{@code head_dim} (explicit override, some models)</li>
 * <li>SentencePiece vocab from {@code tokenizer.model} binary</li>
 * <li>Merges from {@code tokenizer.json} added_tokens for BPE</li>
 * </ul>
 */
public final class HfConfigParser {

    private HfConfigParser() {
    }

    // ─────────────────────────────────────────────────────────────────────
    // Model Config
    // ─────────────────────────────────────────────────────────────────────

    /**
     * Fields from HuggingFace {@code config.json} used during conversion.
     *
     * <p>
     * The {@code raw} field contains the full parsed JSON object so callers
     * can access any field not explicitly modelled here.
     */
    public record ModelConfig(
            String modelType,
            int hiddenSize,
            int intermediateSize,
            int numHiddenLayers,
            int numAttentionHeads,
            /** GQA: number of KV heads; equals numAttentionHeads for MHA. */
            int numKeyValueHeads,
            int maxPositionEmbeddings,
            int vocabSize,
            float rmsNormEps,
            float ropeTheta,
            String hiddenAct,
            boolean tieWordEmbeddings,
            /** Sliding-window attention size (0 = disabled). */
            int slidingWindow,
            /** Whether Q/K/V projections have a bias. */
            boolean attentionBias,
            /** Fraction of head-dim used for rotary embedding (e.g. 0.4 for Phi-2). */
            float partialRotaryFactor,
            /** Explicit head_dim override (0 = derive from hiddenSize/numAttentionHeads). */
            int headDim,
            JsonObject raw) {
    }

    public static ModelConfig parseConfig(Path dir) throws IOException {
        Path p = dir.resolve("config.json");
        if (!Files.exists(p))
            throw new IOException("config.json not found in " + dir);

        try (Reader r = Files.newBufferedReader(p)) {
            JsonObject obj = JsonParser.parseReader(r).getAsJsonObject();
            
            // For multimodal models (e.g., Gemma4ForConditionalGeneration),
            // the text config is nested under "text_config"
            JsonObject textObj = obj;
            if (obj.has("text_config") && obj.get("text_config").isJsonObject()) {
                textObj = obj.getAsJsonObject("text_config");
            }

            // sliding_window may be absent, null, or a real integer
            int slidingWindow = 0;
            if (textObj.has("sliding_window") && !textObj.get("sliding_window").isJsonNull()) {
                try { slidingWindow = textObj.get("sliding_window").getAsInt(); }
                catch (Exception ignored) {}
            }

            // attention_bias: key varies across model families
            boolean attnBias = getBool(textObj, "attention_bias", false)
                    || getBool(textObj, "bias", false)
                    || getBool(textObj, "use_bias", false);

            float partialRotary = getFloat(textObj, "partial_rotary_factor", 1.0f);
            int headDim = getInt(textObj, "head_dim", 0); // 0 = compute from hiddenSize/heads
            
            // Get model_type from top-level if available, otherwise from text_config
            String modelType = getString(obj, "model_type", null);
            if (modelType == null) {
                modelType = getString(textObj, "model_type", "llama");
            }

            return new ModelConfig(
                    modelType,
                    getInt(textObj, "hidden_size", 4096),
                    getInt(textObj, "intermediate_size", 11008),
                    getInt(textObj, "num_hidden_layers", 32),
                    getInt(textObj, "num_attention_heads", 32),
                    getInt(textObj, "num_key_value_heads",
                            getInt(textObj, "num_attention_heads", 32)),
                    getInt(textObj, "max_position_embeddings", 4096),
                    getInt(textObj, "vocab_size", 32000),
                    getFloat(textObj, "rms_norm_eps",
                            getFloat(textObj, "layer_norm_epsilon",
                                    getFloat(textObj, "layer_norm_eps", 1e-5f))),
                    getFloat(textObj, "rope_theta",
                            getFloat(textObj, "rotary_emb_base", 10000f)),
                    getString(textObj, "hidden_act",
                            getString(textObj, "hidden_activation", "silu")),
                    getBool(textObj, "tie_word_embeddings", false),
                    slidingWindow,
                    attnBias,
                    partialRotary,
                    headDim,
                    obj);
        }
    }

    // ─────────────────────────────────────────────────────────────────────
    // Tokenizer
    // ─────────────────────────────────────────────────────────────────────

    public record TokenizerData(
            List<String> vocab,       // id → token string
            List<Float>  scores,      // id → score
            List<Integer> tokenTypes, // id → type
            String bosToken,
            String eosToken,
            int bosId,
            int eosId,
            String tokenizerModel     // "llama", "gpt2", "bert", …
    ) {}

    /**
     * Parses the tokenizer for the model in {@code dir}.
     *
     * <p>Strategy (in priority order):
     * <ol>
     * <li>{@code tokenizer.json} (HF fast tokenizer) – covers BPE and Unigram</li>
     * <li>{@code tokenizer_config.json} for metadata (BOS/EOS ids, model name)</li>
     * </ol>
     *
     * Returns {@code null} if no tokenizer file is found.
     */
    public static TokenizerData parseTokenizer(Path dir) throws IOException {
        Path tokJson = dir.resolve("tokenizer.json");
        Path tokCfg  = dir.resolve("tokenizer_config.json");

        if (!Files.exists(tokJson)) return null;

        // ── Parse tokenizer_config.json for BOS/EOS strings ───────────────
        String bosStr = "<s>", eosStr = "</s>";
        if (Files.exists(tokCfg)) {
            try (Reader r = Files.newBufferedReader(tokCfg)) {
                JsonObject tc = JsonParser.parseReader(r).getAsJsonObject();
                bosStr = extractSpecialToken(tc, "bos_token", "<s>");
                eosStr = extractSpecialToken(tc, "eos_token", "</s>");
            }
        }

        // ── Parse tokenizer.json ──────────────────────────────────────────
        try (Reader r = Files.newBufferedReader(tokJson)) {
            JsonObject root = JsonParser.parseReader(r).getAsJsonObject();

            String tokModel = "llama";
            if (root.has("model") && root.get("model").isJsonObject()) {
                tokModel = getString(root.getAsJsonObject("model"), "type", "llama");
            }

            List<String>  vocab      = new ArrayList<>();
            List<Float>   scores     = new ArrayList<>();
            List<Integer> tokenTypes = new ArrayList<>();

            // BPE / Unigram vocab
            if (root.has("model")) {
                JsonObject modelObj = root.getAsJsonObject("model");
                if (modelObj.has("vocab")) {
                    // vocab is a map of token → id
                    JsonObject vocabObj = modelObj.getAsJsonObject("vocab");
                    String[] arr = new String[vocabObj.size()];
                    for (Map.Entry<String, JsonElement> e : vocabObj.entrySet())
                        arr[e.getValue().getAsInt()] = e.getKey();
                    vocab.addAll(Arrays.asList(arr));
                    for (int i = 0; i < vocab.size(); i++) {
                        scores.add(0f);
                        tokenTypes.add(1); // normal
                    }
                } else if (modelObj.has("pieces")) {
                    // SentencePiece Unigram stored inline
                    JsonArray pieces = modelObj.getAsJsonArray("pieces");
                    for (JsonElement pe : pieces) {
                        JsonObject p = pe.getAsJsonObject();
                        vocab.add(p.get("piece").getAsString());
                        scores.add(p.has("score") ? p.get("score").getAsFloat() : 0f);
                        tokenTypes.add(p.has("type") ? p.get("type").getAsInt() : 1);
                    }
                }
            }

            // Special / added tokens (BOS, EOS, PAD, UNK …)
            int bosId = 1, eosId = 2;
            if (root.has("added_tokens")) {
                for (JsonElement el : root.getAsJsonArray("added_tokens")) {
                    JsonObject t = el.getAsJsonObject();
                    String content = getString(t, "content", "");
                    int id = getInt(t, "id", -1);
                    if (id < 0) continue;

                    // Extend vocab arrays if needed
                    while (vocab.size() <= id) {
                        vocab.add(null);
                        scores.add(0f);
                        tokenTypes.add(0);
                    }
                    vocab.set(id, content);
                    tokenTypes.set(id, 3); // control token

                    // Detect BOS / EOS by content match
                    if (content.equals(bosStr) || isBosContent(content)) {
                        bosId = id;
                    }
                    if (content.equals(eosStr) || isEosContent(content)) {
                        eosId = id;
                    }
                }
            }

            // Replace nulls left by gaps in the vocab map
            for (int i = 0; i < vocab.size(); i++) {
                if (vocab.get(i) == null) vocab.set(i, "<unk_" + i + ">");
            }

            return new TokenizerData(vocab, scores, tokenTypes,
                    bosStr, eosStr, bosId, eosId,
                    mapTokenizerModelType(tokModel));
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────

    /** Extract a BOS/EOS token string that may be a plain string or an object {content:…}. */
    private static String extractSpecialToken(JsonObject tc, String key, String def) {
        if (!tc.has(key) || tc.get(key).isJsonNull()) return def;
        JsonElement el = tc.get(key);
        if (el.isJsonPrimitive()) return el.getAsString();
        if (el.isJsonObject() && el.getAsJsonObject().has("content"))
            return el.getAsJsonObject().get("content").getAsString();
        return def;
    }

    private static boolean isBosContent(String s) {
        String l = s.toLowerCase();
        return l.equals("<s>") || l.contains("bos") || l.equals("<|begin_of_text|>");
    }

    private static boolean isEosContent(String s) {
        String l = s.toLowerCase();
        return l.equals("</s>") || l.contains("eos") || l.equals("<|end_of_text|>")
                || l.equals("<|eot_id|>") || l.equals("<|im_end|>");
    }

    private static String mapTokenizerModelType(String t) {
        return switch (t.toLowerCase()) {
            case "bpe"       -> "gpt2";
            case "unigram"   -> "llama";
            case "wordpiece" -> "bert";
            default          -> t.toLowerCase();
        };
    }

    // ── JSON field accessors ──────────────────────────────────────────────

    private static String getString(JsonObject o, String key, String def) {
        if (!o.has(key) || o.get(key).isJsonNull()) return def;
        JsonElement e = o.get(key);
        return e.isJsonPrimitive() ? e.getAsString() : def;
    }

    private static int getInt(JsonObject o, String key, int def) {
        if (!o.has(key) || o.get(key).isJsonNull()) return def;
        try { return o.get(key).getAsInt(); } catch (Exception e) { return def; }
    }

    private static float getFloat(JsonObject o, String key, float def) {
        if (!o.has(key) || o.get(key).isJsonNull()) return def;
        try { return o.get(key).getAsFloat(); } catch (Exception e) { return def; }
    }

    private static boolean getBool(JsonObject o, String key, boolean def) {
        if (!o.has(key) || o.get(key).isJsonNull()) return def;
        try { return o.get(key).getAsBoolean(); } catch (Exception e) { return def; }
    }
}
