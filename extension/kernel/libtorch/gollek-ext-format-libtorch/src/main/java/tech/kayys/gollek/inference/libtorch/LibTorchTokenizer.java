package tech.kayys.gollek.inference.libtorch;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.enterprise.context.ApplicationScoped;
import jakarta.inject.Inject;
import org.jboss.logging.Logger;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * BPE Tokenizer for LibTorch models.
 * <p>
 * Loads a HuggingFace-compatible {@code tokenizer.json} and performs
 * Byte-Pair Encoding (BPE) tokenization. Supports:
 * <ul>
 * <li>Full BPE encode/decode with merge rules</li>
 * <li>Special token handling (BOS, EOS, PAD, UNK)</li>
 * <li>Pre-tokenization via regex splitting (GPT-2/LLaMA style)</li>
 * <li>Per-model tokenizer caching</li>
 * </ul>
 * <p>
 * Falls back to a character-level tokenizer if no tokenizer.json is found.
 */
@ApplicationScoped
public class LibTorchTokenizer {

    private static final Logger log = Logger.getLogger(LibTorchTokenizer.class);

    /**
     * GPT-2 / LLaMA-style pre-tokenization pattern.
     * Splits on word boundaries, contractions, numbers, and whitespace.
     */
    private static final Pattern PRE_TOKENIZE_PATTERN = Pattern.compile(
            "'(?:[sdmt]|ll|ve|re)|[^\\r\\n\\p{L}\\p{N}]?+\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]++[\\r\\n]*|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+");

    @Inject
    LibTorchProviderConfig config;

    private final Map<String, TokenizerState> tokenizerCache = new ConcurrentHashMap<>();

    /**
     * Tokenize text into token IDs using the model's tokenizer.
     *
     * @param modelId the model identifier (used to locate tokenizer.json)
     * @param text    input text
     * @param addBos  whether to prepend BOS token
     * @return array of token IDs
     */
    public long[] encode(String modelId, String text, boolean addBos) {
        TokenizerState state = getOrLoadTokenizer(modelId);
        List<Long> tokens = new ArrayList<>();

        if (addBos && state.bosTokenId >= 0) {
            tokens.add((long) state.bosTokenId);
        }

        // Pre-tokenize into words
        List<String> words = preTokenize(text);

        for (String word : words) {
            tokens.addAll(bpeEncode(word, state));
        }

        return tokens.stream().mapToLong(Long::longValue).toArray();
    }

    /**
     * Decode token IDs back to text.
     *
     * @param modelId  the model identifier
     * @param tokenIds token IDs to decode
     * @return decoded text
     */
    public String decode(String modelId, long[] tokenIds) {
        TokenizerState state = getOrLoadTokenizer(modelId);
        StringBuilder sb = new StringBuilder();

        for (long tokenId : tokenIds) {
            int id = (int) tokenId;
            // Skip special tokens
            if (id == state.bosTokenId || id == state.eosTokenId || id == state.padTokenId) {
                continue;
            }
            String piece = state.idToToken.getOrDefault(id, "");
            sb.append(piece);
        }

        // Decode byte-level BPE encoding (Ġ → space, etc.)
        return decodeBpeOutput(sb.toString());
    }

    /**
     * Decode a single token ID to its text piece.
     */
    public String decodeToken(String modelId, long tokenId) {
        TokenizerState state = getOrLoadTokenizer(modelId);
        int id = (int) tokenId;
        if (id == state.eosTokenId)
            return "";
        String piece = state.idToToken.getOrDefault(id, "");
        return decodeBpeOutput(piece);
    }

    /**
     * Get the vocabulary size for a model.
     */
    public int vocabSize(String modelId) {
        return getOrLoadTokenizer(modelId).vocabSize;
    }

    /**
     * Get the EOS token ID for a model.
     */
    public int eosTokenId(String modelId) {
        return getOrLoadTokenizer(modelId).eosTokenId;
    }

    // ── Core BPE implementation ──────────────────────────────────────

    private List<Long> bpeEncode(String word, TokenizerState state) {
        if (word.isEmpty())
            return Collections.emptyList();

        // Convert word to byte-level representation
        List<String> symbols = new ArrayList<>();
        for (char c : word.toCharArray()) {
            String byteRep = state.byteEncoder.getOrDefault(c, String.valueOf(c));
            symbols.add(byteRep);
        }

        // Iteratively apply BPE merges
        while (symbols.size() > 1) {
            // Find the best pair (lowest rank in merge rules)
            int bestIdx = -1;
            int bestRank = Integer.MAX_VALUE;

            for (int i = 0; i < symbols.size() - 1; i++) {
                String pair = symbols.get(i) + symbols.get(i + 1);
                Integer rank = state.mergeRanks.get(pair);
                if (rank != null && rank < bestRank) {
                    bestRank = rank;
                    bestIdx = i;
                }
            }

            if (bestIdx < 0)
                break; // No more merges possible

            // Apply the merge
            String merged = symbols.get(bestIdx) + symbols.get(bestIdx + 1);
            symbols.set(bestIdx, merged);
            symbols.remove(bestIdx + 1);
        }

        // Convert symbols to token IDs
        List<Long> ids = new ArrayList<>();
        for (String sym : symbols) {
            Integer id = state.tokenToId.get(sym);
            if (id != null) {
                ids.add((long) id);
            } else {
                // Unknown token — use UNK or encode byte-by-byte
                if (state.unkTokenId >= 0) {
                    ids.add((long) state.unkTokenId);
                } else {
                    // Byte fallback
                    for (byte b : sym.getBytes(java.nio.charset.StandardCharsets.UTF_8)) {
                        String byteToken = String.format("<0x%02X>", b & 0xFF);
                        Integer byteId = state.tokenToId.get(byteToken);
                        if (byteId != null) {
                            ids.add((long) byteId);
                        }
                    }
                }
            }
        }

        return ids;
    }

    private List<String> preTokenize(String text) {
        List<String> words = new ArrayList<>();
        Matcher matcher = PRE_TOKENIZE_PATTERN.matcher(text);
        while (matcher.find()) {
            words.add(matcher.group());
        }
        if (words.isEmpty() && !text.isEmpty()) {
            words.add(text); // Fallback: single token
        }
        return words;
    }

    /**
     * Decode BPE byte-level encoding back to normal text.
     * Handles the GPT-2 byte encoder convention (Ġ = space, Ċ = newline, etc.)
     */
    private String decodeBpeOutput(String bpeText) {
        StringBuilder result = new StringBuilder();
        for (int i = 0; i < bpeText.length(); i++) {
            char c = bpeText.charAt(i);
            if (c == 'Ġ') {
                result.append(' ');
            } else if (c == 'Ċ') {
                result.append('\n');
            } else if (c == 'ĉ') {
                result.append('\t');
            } else {
                result.append(c);
            }
        }
        return result.toString();
    }

    // ── Tokenizer loading ────────────────────────────────────────────

    private TokenizerState getOrLoadTokenizer(String modelId) {
        return tokenizerCache.computeIfAbsent(modelId, this::loadTokenizer);
    }

    private TokenizerState loadTokenizer(String modelId) {
        // Try to find tokenizer.json next to the model
        Path modelBasePath = Path.of(config.model().basePath());
        Path tokenizerPath = modelBasePath.resolve(modelId).resolve("tokenizer.json");

        // Also try parent folder (model might be a single .pt file)
        if (!Files.exists(tokenizerPath)) {
            tokenizerPath = modelBasePath.resolve("tokenizer.json");
        }

        // Try model ID as directory name with stripped extension
        if (!Files.exists(tokenizerPath)) {
            String cleanId = modelId.replaceAll("\\.(pt|pts|pth)$", "");
            tokenizerPath = modelBasePath.resolve(cleanId).resolve("tokenizer.json");
        }

        if (Files.exists(tokenizerPath)) {
            try {
                log.infof("Loading BPE tokenizer from: %s", tokenizerPath);
                return loadFromTokenizerJson(tokenizerPath);
            } catch (Exception e) {
                log.warnf(e, "Failed to load tokenizer.json, falling back to character-level tokenizer");
            }
        }

        log.warnf("No tokenizer.json found for model '%s', using character-level fallback", modelId);
        return createCharLevelTokenizer();
    }

    private TokenizerState loadFromTokenizerJson(Path path) throws IOException {
        ObjectMapper mapper = new ObjectMapper();
        TokenizerJson json = mapper.readValue(path.toFile(), TokenizerJson.class);

        TokenizerState state = new TokenizerState();

        // Load vocabulary
        if (json.model != null && json.model.vocab != null) {
            state.tokenToId = new HashMap<>(json.model.vocab);
            state.idToToken = new HashMap<>();
            json.model.vocab.forEach((token, id) -> state.idToToken.put(id, token));
            state.vocabSize = json.model.vocab.size();
        }

        // Load merge rules
        if (json.model != null && json.model.merges != null) {
            state.mergeRanks = new HashMap<>();
            for (int i = 0; i < json.model.merges.size(); i++) {
                String merge = json.model.merges.get(i);
                // Merge format: "token1 token2" → merged as "token1token2"
                String[] parts = merge.split(" ", 2);
                if (parts.length == 2) {
                    state.mergeRanks.put(parts[0] + parts[1], i);
                }
            }
        }

        // Load special tokens
        if (json.addedTokens != null) {
            for (TokenizerJson.AddedToken at : json.addedTokens) {
                state.tokenToId.put(at.content, at.id);
                state.idToToken.put(at.id, at.content);

                if (at.content.contains("bos") || at.content.equals("<s>")) {
                    state.bosTokenId = at.id;
                } else if (at.content.contains("eos") || at.content.equals("</s>")) {
                    state.eosTokenId = at.id;
                } else if (at.content.contains("pad") || at.content.equals("<pad>")) {
                    state.padTokenId = at.id;
                } else if (at.content.contains("unk") || at.content.equals("<unk>")) {
                    state.unkTokenId = at.id;
                }
            }
        }

        // Build byte encoder (GPT-2 style)
        state.byteEncoder = buildByteEncoder();

        log.infof("Loaded tokenizer: vocab=%d, merges=%d, bos=%d, eos=%d",
                state.vocabSize,
                state.mergeRanks != null ? state.mergeRanks.size() : 0,
                state.bosTokenId, state.eosTokenId);

        return state;
    }

    /**
     * Character-level fallback tokenizer.
     * Maps each Unicode codepoint to a token ID.
     */
    private TokenizerState createCharLevelTokenizer() {
        TokenizerState state = new TokenizerState();
        state.tokenToId = new HashMap<>();
        state.idToToken = new HashMap<>();
        state.mergeRanks = new HashMap<>();
        state.byteEncoder = new HashMap<>();

        // Reserve special tokens
        state.bosTokenId = 1;
        state.eosTokenId = 2;
        state.padTokenId = 0;
        state.unkTokenId = 3;

        state.idToToken.put(0, "<pad>");
        state.idToToken.put(1, "<s>");
        state.idToToken.put(2, "</s>");
        state.idToToken.put(3, "<unk>");
        state.tokenToId.put("<pad>", 0);
        state.tokenToId.put("<s>", 1);
        state.tokenToId.put("</s>", 2);
        state.tokenToId.put("<unk>", 3);

        // Map printable ASCII + extended to IDs starting at 4
        int nextId = 4;
        for (int i = 0; i < 256; i++) {
            String ch = String.valueOf((char) i);
            state.tokenToId.put(ch, nextId);
            state.idToToken.put(nextId, ch);
            state.byteEncoder.put((char) i, ch);
            nextId++;
        }
        state.vocabSize = nextId;

        log.info("Using character-level fallback tokenizer (vocab=260)");
        return state;
    }

    /**
     * Build the GPT-2 byte encoder mapping.
     * Maps bytes 0-255 to printable Unicode characters to avoid
     * whitespace/control characters in the BPE vocabulary.
     */
    private Map<Character, String> buildByteEncoder() {
        Map<Character, String> encoder = new HashMap<>();
        // Printable ASCII range + extended
        List<Integer> bs = new ArrayList<>();
        for (int i = '!'; i <= '~'; i++)
            bs.add(i);
        for (int i = '¡'; i <= '¬'; i++)
            bs.add(i);
        for (int i = '®'; i <= 'ÿ'; i++)
            bs.add(i);

        List<Integer> cs = new ArrayList<>(bs);
        int n = 0;
        for (int b = 0; b < 256; b++) {
            if (!bs.contains(b)) {
                bs.add(b);
                cs.add(256 + n);
                n++;
            }
        }

        for (int i = 0; i < bs.size(); i++) {
            encoder.put((char) bs.get(i).intValue(), String.valueOf((char) cs.get(i).intValue()));
        }
        return encoder;
    }

    // ── Internal state ───────────────────────────────────────────────

    static class TokenizerState {
        Map<String, Integer> tokenToId = new HashMap<>();
        Map<Integer, String> idToToken = new HashMap<>();
        Map<String, Integer> mergeRanks = new HashMap<>();
        Map<Character, String> byteEncoder = new HashMap<>();
        int vocabSize;
        int bosTokenId = -1;
        int eosTokenId = -1;
        int padTokenId = -1;
        int unkTokenId = -1;
    }

    // ── JSON model ───────────────────────────────────────────────────

    @JsonIgnoreProperties(ignoreUnknown = true)
    static class TokenizerJson {
        @JsonProperty("model")
        public Model model;

        @JsonProperty("added_tokens")
        public List<AddedToken> addedTokens;

        @JsonIgnoreProperties(ignoreUnknown = true)
        static class Model {
            @JsonProperty("type")
            public String type;

            @JsonProperty("vocab")
            public Map<String, Integer> vocab;

            @JsonProperty("merges")
            public List<String> merges;
        }

        @JsonIgnoreProperties(ignoreUnknown = true)
        static class AddedToken {
            @JsonProperty("id")
            public int id;

            @JsonProperty("content")
            public String content;

            @JsonProperty("special")
            public boolean special;
        }
    }
}
