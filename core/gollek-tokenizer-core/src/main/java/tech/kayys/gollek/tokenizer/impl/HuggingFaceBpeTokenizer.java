package tech.kayys.gollek.tokenizer.impl;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.jboss.logging.Logger;
import tech.kayys.gollek.tokenizer.spi.DecodeOptions;
import tech.kayys.gollek.tokenizer.spi.EncodeOptions;
import tech.kayys.gollek.tokenizer.spi.PreTokenizer;
import tech.kayys.gollek.tokenizer.spi.Tokenizer;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.Locale;

/**
 * BPE Tokenizer implementation for HuggingFace-compatible models.
 *
 * <p>Implementation logic moved from the original HuggingFaceTokenizer to comply
 * with the new modular SPI. Uses a PreTokenizer for initial text splitting.
 */
public class HuggingFaceBpeTokenizer implements Tokenizer {

    private static final Logger log = Logger.getLogger(HuggingFaceBpeTokenizer.class);

    private final TokenizerState state;
    private final PreTokenizer preTokenizer;
    private final Map<Character, Integer> charToByte = new HashMap<>();
    private final boolean useByteEncoder;
    private final boolean decodeWhitespace;

    public HuggingFaceBpeTokenizer(TokenizerState state, PreTokenizer preTokenizer,
            boolean useByteEncoder, boolean decodeWhitespace) {
        this.state = state;
        this.preTokenizer = preTokenizer;
        this.useByteEncoder = useByteEncoder;
        this.decodeWhitespace = decodeWhitespace;
        // Initialize charToByte mapping
        if (useByteEncoder && state.byteEncoder != null) {
            state.byteEncoder.forEach((b, s) -> charToByte.put(s.charAt(0), (int) b));
        }
    }

    public static HuggingFaceBpeTokenizer load(Path tokenizerPath, PreTokenizer preTokenizer) throws IOException {
        return load(tokenizerPath, preTokenizer, true, false);
    }

    public static HuggingFaceBpeTokenizer load(Path vocabPath, Path mergesPath, PreTokenizer preTokenizer,
            boolean useByteEncoder, boolean decodeWhitespace) throws IOException {
        ObjectMapper mapper = new ObjectMapper();
        TokenizerState state = new TokenizerState();

        // Load Vocab
        JsonNode vocabNode = mapper.readTree(vocabPath.toFile());
        if (vocabNode.isObject()) {
            state.tokenToId = new HashMap<>();
            state.idToToken = new HashMap<>();
            vocabNode.fields().forEachRemaining(e -> {
                int id = e.getValue().asInt();
                state.tokenToId.put(e.getKey(), id);
                state.idToToken.put(id, e.getKey());
            });
            state.vocabSize = state.tokenToId.size();
        }

        // Load Merges
        state.mergeRanks = new HashMap<>();
        List<String> lines = Files.readAllLines(mergesPath);
        int idx = 0;
        for (String line : lines) {
            line = line.trim();
            if (line.isEmpty() || line.startsWith("#")) continue;
            state.mergeRanks.put(line.replace(" ", ""), idx++);
        }

        if (useByteEncoder) {
            state.byteEncoder = buildByteEncoder();
        }
        return new HuggingFaceBpeTokenizer(state, preTokenizer, useByteEncoder, decodeWhitespace);
    }

    public static HuggingFaceBpeTokenizer load(Path tokenizerPath, PreTokenizer preTokenizer,
            boolean useByteEncoder, boolean decodeWhitespace) throws IOException {
        ObjectMapper mapper = new ObjectMapper();
        JsonNode root = mapper.readTree(tokenizerPath.toFile());

        TokenizerState state = new TokenizerState();

        JsonNode modelNode = root.path("model");
        JsonNode vocabNode = modelNode.path("vocab");
        if (vocabNode.isObject()) {
            state.tokenToId = new HashMap<>();
            state.idToToken = new HashMap<>();
            vocabNode.fields().forEachRemaining(e -> {
                int id = e.getValue().asInt();
                state.tokenToId.put(e.getKey(), id);
                state.idToToken.put(id, e.getKey());
            });
            state.vocabSize = state.tokenToId.size();
        }

        JsonNode mergesNode = modelNode.path("merges");
        if (mergesNode.isArray()) {
            state.mergeRanks = new HashMap<>();
            int idx = 0;
            for (JsonNode merge : mergesNode) {
                String left;
                String right;
                if (merge.isTextual()) {
                    String[] parts = merge.asText().split(" ", 2);
                    if (parts.length != 2) {
                        idx++;
                        continue;
                    }
                    left = parts[0];
                    right = parts[1];
                } else if (merge.isArray() && merge.size() >= 2) {
                    left = merge.get(0).asText();
                    right = merge.get(1).asText();
                } else {
                    idx++;
                    continue;
                }
                state.mergeRanks.put(left + right, idx);
                idx++;
            }
        }

        JsonNode addedTokensNode = root.path("added_tokens");
        if (addedTokensNode.isArray()) {
            state.specialTokens = new HashMap<>();
            for (JsonNode at : addedTokensNode) {
                if (!at.hasNonNull("id") || !at.hasNonNull("content")) {
                    continue;
                }
                int id = at.get("id").asInt();
                String content = at.get("content").asText();
                state.tokenToId.put(content, id);
                state.idToToken.put(id, content);
                state.specialTokens.put(content, id);

                String lower = content.toLowerCase(Locale.ROOT);
                if (lower.contains("bos") || lower.equals("<s>") || lower.equals("<|im_start|>") || lower.equals("<|begin_of_text|>")) {
                    if (state.bosTokenId < 0) state.bosTokenId = id;
                } else if (lower.contains("eos") || lower.equals("</s>") || lower.equals("<|im_end|>") || lower.equals("<|endoftext|>") || lower.equals("<|eot_id|>")) {
                    if (state.eosTokenId < 0) state.eosTokenId = id;
                } else if (lower.contains("pad") || lower.equals("<pad>")) {
                    state.padTokenId = id;
                } else if (lower.contains("unk") || lower.equals("<unk>")) {
                    state.unkTokenId = id;
                }
            }
        }

        if (useByteEncoder) {
            state.byteEncoder = buildByteEncoder();
        }
        return new HuggingFaceBpeTokenizer(state, preTokenizer, useByteEncoder, decodeWhitespace);
    }

    @Override
    public long[] encode(String text, EncodeOptions options) {
        List<Long> tokens = new ArrayList<>();

        if (options.addBos && state.bosTokenId >= 0) {
            tokens.add((long) state.bosTokenId);
        }

        // Special token priority encoding: match added_tokens exactly before BPE
        if (state.specialTokens != null && !state.specialTokens.isEmpty()) {
            List<Object> parts = parseSpecialTokens(text);
            for (Object part : parts) {
                if (part instanceof Integer id) {
                    tokens.add((long) id);
                } else if (part instanceof String s) {
                    List<String> words = preTokenizer.split(s);
                    for (String word : words) {
                        tokens.addAll(bpeEncode(word));
                    }
                }
            }
        } else {
            List<String> words = preTokenizer.split(text);
            for (String word : words) {
                tokens.addAll(bpeEncode(word));
            }
        }

        if (options.addEos && state.eosTokenId >= 0) {
            tokens.add((long) state.eosTokenId);
        }

        long[] result = tokens.stream().mapToLong(Long::longValue).toArray();
        if (log.isDebugEnabled()) {
            log.debugf("Encoded tokens: %s", Arrays.toString(result));
            for (long id : result) {
                log.debugf("  ID %d -> [%s]", id, state.idToToken.get((int)id));
            }
        }
        System.out.println("[DIAG-TOKENIZER] Input: \"" + text + "\" -> " + Arrays.toString(result));
        return result;
    }

    private List<Object> parseSpecialTokens(String text) {
        List<Object> result = new ArrayList<>();
        int current = 0;
        
        while (current < text.length()) {
            String bestMatch = null;
            int bestId = -1;
            
            for (Map.Entry<String, Integer> entry : state.specialTokens.entrySet()) {
                String key = entry.getKey();
                if (text.startsWith(key, current)) {
                    if (bestMatch == null || key.length() > bestMatch.length()) {
                        bestMatch = key;
                        bestId = entry.getValue();
                    }
                }
            }
            
            if (bestMatch != null) {
                current += bestMatch.length();
                result.add(bestId);
            } else {
                // Find next special token start
                int nextStart = text.length();
                for (String key : state.specialTokens.keySet()) {
                    int idx = text.indexOf(key, current + 1);
                    if (idx != -1 && idx < nextStart) {
                        nextStart = idx;
                    }
                }
                result.add(text.substring(current, nextStart));
                current = nextStart;
            }
        }
        return result;
    }

    @Override
    public String decode(long[] tokens, DecodeOptions options) {
        StringBuilder sb = new StringBuilder();
        for (long tokenId : tokens) {
            int id = (int) tokenId;
            if (options.skipSpecialTokens && isSpecialToken(id)) {
                continue;
            }
            String piece = state.idToToken.getOrDefault(id, "");
            sb.append(piece);
        }
        String decoded = decodeBpeOutput(sb.toString());
        if (decodeWhitespace) {
            decoded = decoded.replace('▁', ' ');
        }
        return decoded;
    }

    private boolean isSpecialToken(int id) {
        return id == state.bosTokenId || id == state.eosTokenId || id == state.padTokenId || id == state.unkTokenId;
    }

    @Override public int vocabSize() { return state.vocabSize; }
    @Override public int bosTokenId() { return state.bosTokenId; }
    @Override public int eosTokenId() { return state.eosTokenId; }
    @Override public int padTokenId() { return state.padTokenId; }

    private List<Long> bpeEncode(String word) {
        if (word.isEmpty()) return Collections.emptyList();

        List<String> symbols = new ArrayList<>();
        for (char c : word.toCharArray()) {
            if (useByteEncoder && state.byteEncoder != null) {
                String byteRep = state.byteEncoder.getOrDefault(c, String.valueOf(c));
                symbols.add(byteRep);
            } else {
                symbols.add(String.valueOf(c));
            }
        }

        while (symbols.size() > 1) {
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

            if (bestIdx < 0) break;

            String merged = symbols.get(bestIdx) + symbols.get(bestIdx + 1);
            symbols.set(bestIdx, merged);
            symbols.remove(bestIdx + 1);
        }

        List<Long> ids = new ArrayList<>();
        for (String sym : symbols) {
            Integer id = state.tokenToId.get(sym);
            if (id != null) {
                ids.add((long) id);
            } else {
                if (state.unkTokenId >= 0) {
                    ids.add((long) state.unkTokenId);
                } else {
                    // BBPE Fallback: try to find byte tokens directly
                    byte[] bytes = sym.getBytes(java.nio.charset.StandardCharsets.UTF_8);
                    for (byte b : bytes) {
                        int unsigned = b & 0xFF;
                        // For BBPE models like Qwen, bytes are mapped back via byteEncoder characters
                        String bChar = state.byteEncoder.get((char)unsigned);
                        Integer bId = state.tokenToId.get(bChar);
                        if (bId != null) {
                            ids.add((long) bId);
                        }
                    }
                }
            }
        }
        return ids;
    }

    private String decodeBpeOutput(String bpeText) {
        if (bpeText == null || bpeText.isEmpty()) return "";

        byte[] bytes = new byte[bpeText.length() * 3];
        int byteIdx = 0;
        
        for (int i = 0; i < bpeText.length(); i++) {
            char c = bpeText.charAt(i);

            // Handle hex-encoded tokens
            if (c == '<' && i + 5 < bpeText.length() && bpeText.charAt(i + 1) == '0' && bpeText.charAt(i + 2) == 'x' && bpeText.charAt(i + 5) == '>') {
                try {
                    String hex = bpeText.substring(i + 3, i + 5);
                    int b = Integer.parseInt(hex, 16);
                    if (byteIdx + 1 > bytes.length) bytes = Arrays.copyOf(bytes, bytes.length * 2);
                    bytes[byteIdx++] = (byte) b;
                    i += 5;
                    continue;
                } catch (NumberFormatException ignored) {}
            }

            Integer b = charToByte.get(c);
            byte[] toAppend;
            if (b != null) {
                toAppend = new byte[]{(byte) b.intValue()};
            } else {
                toAppend = String.valueOf(c).getBytes(java.nio.charset.StandardCharsets.UTF_8);
            }

            if (byteIdx + toAppend.length > bytes.length) {
                bytes = Arrays.copyOf(bytes, Math.max(bytes.length * 2, byteIdx + toAppend.length));
            }
            System.arraycopy(toAppend, 0, bytes, byteIdx, toAppend.length);
            byteIdx += toAppend.length;
        }

        return new String(bytes, 0, byteIdx, java.nio.charset.StandardCharsets.UTF_8);
    }

    private static Map<Character, String> buildByteEncoder() {
        Map<Character, String> encoder = new HashMap<>();
        List<Integer> bs = new ArrayList<>();
        for (int i = '!'; i <= '~'; i++) bs.add(i);
        for (int i = '¡'; i <= '¬'; i++) bs.add(i);
        for (int i = '®'; i <= 'ÿ'; i++) bs.add(i);

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

    public static class TokenizerState {
        public Map<String, Integer> tokenToId = new HashMap<>();
        public Map<Integer, String> idToToken = new HashMap<>();
        public Map<String, Integer> mergeRanks = new HashMap<>();
        public Map<Character, String> byteEncoder = new HashMap<>();
        public Map<String, Integer> specialTokens = new HashMap<>();
        public int vocabSize;
        public int bosTokenId = -1;
        public int eosTokenId = -1;
        public int padTokenId = -1;
        public int unkTokenId = -1;
    }

    @JsonIgnoreProperties(ignoreUnknown = true)
    static class TokenizerJson {
        @JsonProperty("model") public Model model;
        @JsonProperty("added_tokens") public List<AddedToken> addedTokens;

        @JsonIgnoreProperties(ignoreUnknown = true)
        static class Model {
            @JsonProperty("vocab") public Map<String, Integer> vocab;
            @JsonProperty("merges") public List<String> merges;
        }

        @JsonIgnoreProperties(ignoreUnknown = true)
        static class AddedToken {
            @JsonProperty("id") public int id;
            @JsonProperty("content") public String content;
        }
    }
}
