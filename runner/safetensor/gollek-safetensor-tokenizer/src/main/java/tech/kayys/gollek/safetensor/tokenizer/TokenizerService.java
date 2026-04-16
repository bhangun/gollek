/*
 * Gollek Inference Engine — SafeTensor Module
 * Copyright (c) 2026 Kayys.tech
 * SPDX-License-Identifier: Apache-2.0
 *
 * TokenizerService.java  +  HuggingFaceTokenizer
 * ──────────────────────────────────────────────
 * Loads and operates HuggingFace "fast tokenizer" format (tokenizer.json).
 *
 * What is tokenizer.json?
 * ═══════════════════════
 * Every HuggingFace model ships a tokenizer.json alongside its weights.
 * It encodes the full tokenization pipeline in a single file:
 *   - model: BPE / WordPiece / Unigram vocabulary + merge rules
 *   - normalizer: NFD / NFC / lowercase / Bert-style strip
 *   - pre_tokenizer: byte-level BPE / whitespace / metaspace
 *   - post_processor: add BOS/EOS/[SEP]/[CLS] tokens
 *   - decoder: reverse the encoding, handle byte-level mappings
 *   - added_tokens: special tokens like <|im_start|>, <|endoftext|>
 *
 * tokenizer_config.json
 * ═════════════════════
 * Provides the chat template (Jinja2 macro) and BOS/EOS strings.
 * We parse a simplified version — full Jinja2 is not needed for the
 * common LLaMA / ChatML / Mistral templates.
 *
 * Current implementation strategy
 * ════════════════════════════════
 * Full BPE re-implementation in Java is non-trivial.  We use two approaches:
 *
 * A) JTOKENIZER bridge (preferred, zero-JNI):
 *    The HuggingFace Tokenizers library has a Java port
 *    (https://github.com/nicksyostudy/jtokenizers or similar) that reads
 *    tokenizer.json natively.  We wrap it behind the HuggingFaceTokenizer
 *    interface so the rest of the engine is not coupled to the library.
 *
 * B) JNI bridge to tokenizers.so (fallback):
 *    The official Rust tokenizers library can be called via JNI.
 *    A thin FFM wrapper over tokenizers_binding.h is scaffolded but
 *    not yet activated.
 *
 * Both strategies implement the same HuggingFaceTokenizer interface.
 */
package tech.kayys.gollek.safetensor.tokenizer;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import tech.kayys.gollek.tokenizer.runtime.TokenizerFactory;
import tech.kayys.gollek.tokenizer.spi.Tokenizer;

import org.jboss.logging.Logger;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

/**
 * CDI service that loads and caches {@link HuggingFaceTokenizer} instances.
 */
@ApplicationScoped
public class TokenizerService {

    private static final Logger log = Logger.getLogger(TokenizerService.class);

    @Inject
    ObjectMapper objectMapper;

    /** Cache: model-dir absolute path → loaded tokenizer. */
    private final Map<String, Tokenizer> cache = new java.util.concurrent.ConcurrentHashMap<>();

    /**
     * Load the tokenizer from a model directory.
     *
     * @param modelDir directory containing tokenizer.json
     * @return the loaded tokenizer
     * @throws IOException if the tokenizer files cannot be read
     */
    public Tokenizer load(Path modelDir) throws IOException {
        String key = modelDir.toAbsolutePath().normalize().toString();
        if (cache.containsKey(key)) {
            return cache.get(key);
        }

        Path tokenizerJson = modelDir.resolve("tokenizer.json");
        if (!Files.exists(tokenizerJson)) {
            throw new IOException("tokenizer.json not found in: " + modelDir);
        }

        Tokenizer tokenizer = TokenizerFactory.load(modelDir, null);
        cache.put(key, tokenizer);
        return tokenizer;
    }

    // JSON DTOs for other purposes (if needed)

    @JsonIgnoreProperties(ignoreUnknown = true)
    static final class TokenizerConfig {
        @JsonProperty("bos_token")
        Object bosToken;
        @JsonProperty("eos_token")
        Object eosToken;
        @JsonProperty("pad_token")
        Object padToken;
        @JsonProperty("unk_token")
        Object unkToken;
        @JsonProperty("chat_template")
        String chatTemplate;
        @JsonProperty("add_bos_token")
        Boolean addBosToken;
        @JsonProperty("add_eos_token")
        Boolean addEosToken;
        @JsonProperty("model_max_length")
        Integer modelMaxLength;
    }
}
