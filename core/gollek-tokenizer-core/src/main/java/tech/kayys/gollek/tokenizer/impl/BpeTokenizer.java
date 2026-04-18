package tech.kayys.gollek.tokenizer.impl;

import tech.kayys.gollek.tokenizer.spi.DecodeOptions;
import tech.kayys.gollek.tokenizer.spi.EncodeOptions;
import tech.kayys.gollek.tokenizer.spi.Tokenizer;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Byte-Pair Encoding (BPE) Tokenizer implementation.
 * Supports GPT-2 (byte-level) and Tiktoken (raw UTF-8) styles.
 */
public final class BpeTokenizer implements Tokenizer {

    private final Map<String, Integer> tokenToId;
    private final Map<Integer, String> idToToken;
    private final Map<String, Integer> mergeRanks;
    private final Map<Character, String> byteEncoder;
    private final Map<String, Byte> byteDecoder;
    private final boolean rawUtf8Vocab;
    private final Map<String, Integer> specialTokens;
    private final int unkTokenId;
    private final int bosTokenId;
    private final int eosTokenId;
    private final int padTokenId;

    // Standard GPT-2 pre-tokenization regex
    private static final Pattern GPT2_REGEX = Pattern.compile("'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+");

    public BpeTokenizer(
            Map<String, Integer> tokenToId,
            Map<Integer, String> idToToken,
            Map<String, Integer> mergeRanks,
            boolean rawUtf8Vocab,
            int unkTokenId,
            int bosTokenId,
            int eosTokenId,
            int padTokenId,
            Map<String, Integer> specialTokens
    ) {
        this.tokenToId = tokenToId;
        this.idToToken = idToToken;
        this.mergeRanks = mergeRanks;
        this.rawUtf8Vocab = rawUtf8Vocab;
        this.unkTokenId = unkTokenId;
        this.bosTokenId = bosTokenId;
        this.eosTokenId = eosTokenId;
        this.padTokenId = padTokenId;
        this.specialTokens = specialTokens != null ? specialTokens : new HashMap<>();
        this.byteEncoder = buildByteEncoder();
        this.byteDecoder = new HashMap<>();
        for (Map.Entry<Character, String> e : byteEncoder.entrySet()) {
            this.byteDecoder.put(e.getValue(), (byte) (int) e.getKey());
        }
    }

    /**
     * Legacy constructor for existing loaders.
     */
    public BpeTokenizer(
            Map<String, Integer> tokenToId,
            Map<Integer, String> idToToken,
            Map<String, Integer> mergeRanks,
            Map<Character, String> byteEncoder,
            int bosId,
            int eosId,
            int padId,
            int unkId,
            Object preTokenizer // Ignored for now, using default GPT-2 regex
    ) {
        this.tokenToId = tokenToId;
        this.idToToken = idToToken;
        this.mergeRanks = mergeRanks;
        this.rawUtf8Vocab = false;
        this.unkTokenId = unkId;
        this.bosTokenId = bosId;
        this.eosTokenId = eosId;
        this.padTokenId = padId;
        this.specialTokens = new HashMap<>();
        this.byteEncoder = byteEncoder;
        this.byteDecoder = new HashMap<>();
        for (Map.Entry<Character, String> e : byteEncoder.entrySet()) {
            this.byteDecoder.put(e.getValue(), (byte) (int) e.getKey());
        }
    }

    private Map<Character, String> buildByteEncoder() {
        Map<Character, String> map = new HashMap<>();
        List<Integer> bs = new ArrayList<>();
        for (int i = '!'; i <= '~'; i++) bs.add(i);
        for (int i = '¡'; i <= '¬'; i++) bs.add(i);
        for (int i = '®'; i <= 'ÿ'; i++) bs.add(i);

        int n = 0;
        for (int b = 0; b < 256; b++) {
            if (!bs.contains(b)) {
                bs.add(b);
                map.put((char) b, String.valueOf((char) (256 + n++)));
            } else {
                map.put((char) b, String.valueOf((char) b));
            }
        }
        return map;
    }

    @Override
    public long[] encode(String text, EncodeOptions options) {
        if (text == null) return new long[0];
        
        List<Long> allTokens = new ArrayList<>();
        
        // 1. Find special tokens first
        if (specialTokens.isEmpty()) {
            encodeBasic(text, allTokens);
        } else {
            // Build pattern for special tokens
            StringBuilder sb = new StringBuilder();
            List<String> sortedSpecials = new ArrayList<>(specialTokens.keySet());
            sortedSpecials.sort((a, b) -> Integer.compare(b.length(), a.length()));
            for (int i = 0; i < sortedSpecials.size(); i++) {
                if (i > 0) sb.append("|");
                sb.append(Pattern.quote(sortedSpecials.get(i)));
            }
            Pattern specialPattern = Pattern.compile(sb.toString());
            Matcher m = specialPattern.matcher(text);

            int lastEnd = 0;
            while (m.find()) {
                if (m.start() > lastEnd) {
                    encodeBasic(text.substring(lastEnd, m.start()), allTokens);
                }
                allTokens.add((long) specialTokens.get(m.group()));
                lastEnd = m.end();
            }
            if (lastEnd < text.length()) {
                encodeBasic(text.substring(lastEnd), allTokens);
            }
        }

        long[] result = new long[allTokens.size()];
        for (int i = 0; i < allTokens.size(); i++) result[i] = allTokens.get(i);
        return result;
    }

    private void encodeBasic(String text, List<Long> allTokens) {
        Matcher matcher = GPT2_REGEX.matcher(text);
        while (matcher.find()) {
            String segment = matcher.group();
            allTokens.addAll(bpeEncode(segment));
        }
    }

    private List<Long> bpeEncode(String word) {
        List<String> symbols = new ArrayList<>();

        if (rawUtf8Vocab) {
            byte[] bytes = word.getBytes(StandardCharsets.UTF_8);
            for (byte b : bytes) {
                String byteStr = new String(new byte[]{b}, StandardCharsets.ISO_8859_1);
                symbols.add(byteStr);
            }
        } else {
            byte[] bytes = word.getBytes(StandardCharsets.UTF_8);
            for (byte b : bytes) {
                char c = (char) (b & 0xFF);
                symbols.add(byteEncoder.getOrDefault(c, String.valueOf(c)));
            }
        }

        while (symbols.size() > 1) {
            int bestIdx = -1;
            int bestRank = Integer.MAX_VALUE;

            for (int i = 0; i < symbols.size() - 1; i++) {
                String s1 = symbols.get(i);
                String s2 = symbols.get(i + 1);
                
                Integer rank = null;
                if (!mergeRanks.isEmpty()) {
                    rank = mergeRanks.get(s1 + " " + s2);
                } else {
                    rank = tokenToId.get(s1 + s2);
                }

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
            Integer id = tokenToId.get(sym);
            if (id != null) {
                ids.add((long) id);
            } else if (unkTokenId >= 0) {
                ids.add((long) unkTokenId);
            }
        }
        return ids;
    }

    @Override
    public String decode(long[] tokens, DecodeOptions options) {
        if (tokens == null || tokens.length == 0) return "";
        
        // Accumulate all bytes first to correctly handle multi-token UTF-8 characters
        java.io.ByteArrayOutputStream baos = new java.io.ByteArrayOutputStream();
        for (long id : tokens) {
            String tok = idToToken.get((int) id);
            if (tok != null) {
                decodeToStream(tok, baos);
            }
        }
        return baos.toString(StandardCharsets.UTF_8);
    }

    private void decodeToStream(String text, java.io.ByteArrayOutputStream baos) {
        if (text == null || text.isEmpty()) return;
        
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            
            // 1. Handle common remapping markers (GPT-2, SentencePiece, Tiktoken)
            if (c == '\u0120' || c == '\u00A0' || c == '\u2581') {
                baos.write(32); // Space
                continue;
            }
            if (c == '\u010A') {
                baos.write(10); // Newline
                continue;
            }

            // 2. Try byte decoder for GPT-2 style byte-level mappings
            Byte b = byteDecoder.get(String.valueOf(c));
            if (b != null) {
                baos.write(b & 0xFF);
                continue;
            } 
            
            // 3. Fallback: if it's ASCII/Extended ASCII part of a raw vocab
            if (c < 256) {
                baos.write(c & 0xFF);
            } else {
                // 4. Multi-byte character in rawUtf8Vocab mode
                byte[] literal = String.valueOf(c).getBytes(StandardCharsets.UTF_8);
                baos.writeBytes(literal);
            }
        }
    }

    @Override
    public int vocabSize() { return tokenToId.size(); }

    @Override
    public int bosTokenId() { return bosTokenId; }

    @Override
    public int eosTokenId() { return eosTokenId; }

    @Override
    public int padTokenId() { return padTokenId; }
}