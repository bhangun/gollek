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
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.PriorityQueue;

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
    private final boolean sentencePieceLikeWhitespace;
    private final Map<String, Integer> specialTokens;
    private final int unkTokenId;
    private final int bosTokenId;
    private final int eosTokenId;
    private final int padTokenId;
    private final ExecutorService encoderExecutor = Executors.newVirtualThreadPerTaskExecutor();

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
        this.sentencePieceLikeWhitespace = detectSentencePieceWhitespace(tokenToId, rawUtf8Vocab);
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
        this.sentencePieceLikeWhitespace = false;
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
        
        // 1. Find special tokens
        if (specialTokens.isEmpty()) {
            allTokens.addAll(encodeParallel(text));
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
                    allTokens.addAll(encodeParallel(text.substring(lastEnd, m.start())));
                }
                allTokens.add((long) specialTokens.get(m.group()));
                lastEnd = m.end();
            }
            if (lastEnd < text.length()) {
                allTokens.addAll(encodeParallel(text.substring(lastEnd)));
            }
        }

        long[] result = new long[allTokens.size()];
        for (int i = 0; i < allTokens.size(); i++) result[i] = allTokens.get(i);
        return result;
    }

    private List<Long> encodeParallel(String text) {
        List<String> segments = new ArrayList<>();
        Matcher matcher = GPT2_REGEX.matcher(text);
        while (matcher.find()) {
            segments.add(matcher.group());
        }

        if (segments.size() <= 1) {
            return segments.isEmpty() ? List.of() : bpeEncode(segments.get(0));
        }

        try {
            List<Future<List<Long>>> futures = encoderExecutor.invokeAll(
                segments.stream().map(s -> (java.util.concurrent.Callable<List<Long>>) () -> bpeEncode(s)).toList()
            );
            List<Long> result = new ArrayList<>();
            for (Future<List<Long>> f : futures) {
                result.addAll(f.get());
            }
            return result;
        } catch (Exception e) {
            // Fallback
            List<Long> result = new ArrayList<>();
            for (String s : segments) result.addAll(bpeEncode(s));
            return result;
        }
    }

    private List<Long> bpeEncode(String word) {
        if (word.isEmpty()) return List.of();

        if (rawUtf8Vocab && sentencePieceLikeWhitespace && word.indexOf(' ') >= 0) {
            word = word.replace(' ', '\u2581');
        }

        // Initial tokens from raw bytes
        List<Symbol> symbols = new ArrayList<>();
        if (rawUtf8Vocab) {
            for (int i = 0; i < word.length();) {
                int codePoint = word.codePointAt(i);
                symbols.add(new Symbol(new String(Character.toChars(codePoint))));
                i += Character.charCount(codePoint);
            }
        } else {
            byte[] bytes = word.getBytes(StandardCharsets.UTF_8);
            for (byte b : bytes) {
                char c = (char) (b & 0xFF);
                symbols.add(new Symbol(byteEncoder.getOrDefault(c, String.valueOf(c))));
            }
        }

        if (symbols.size() <= 1) {
            return getFinalIds(symbols);
        }

        // Link segments
        for (int i = 0; i < symbols.size(); i++) {
            if (i > 0) symbols.get(i).prev = symbols.get(i - 1);
            if (i < symbols.size() - 1) symbols.get(i).next = symbols.get(i + 1);
        }

        // Priority Queue for merges
        PriorityQueue<Merge> pq = new PriorityQueue<>();
        for (int i = 0; i < symbols.size() - 1; i++) {
            addMerge(pq, symbols.get(i), symbols.get(i + 1));
        }

        while (!pq.isEmpty()) {
            Merge top = pq.poll();
            if (top.s1.deleted || top.s2.deleted) continue;
            if (top.s1.next != top.s2) continue;

            // Perform merge
            String mergedText = top.s1.text + top.s2.text;
            top.s1.text = mergedText;
            top.s2.deleted = true;

            // Update links
            Symbol s3 = top.s2.next;
            top.s1.next = s3;
            if (s3 != null) s3.prev = top.s1;

            // Add new possible merges
            if (top.s1.prev != null) addMerge(pq, top.s1.prev, top.s1);
            if (top.s1.next != null) addMerge(pq, top.s1, top.s1.next);
        }

        return getFinalIds(symbols);
    }

    private void addMerge(PriorityQueue<Merge> pq, Symbol s1, Symbol s2) {
        String pair = s1.text + " " + s2.text;
        Integer rank;
        if (!mergeRanks.isEmpty()) {
            rank = mergeRanks.get(pair);
        } else {
            rank = tokenToId.get(s1.text + s2.text);
        }
        if (rank != null) {
            pq.add(new Merge(s1, s2, rank));
        }
    }

    private List<Long> getFinalIds(List<Symbol> symbols) {
        List<Long> ids = new ArrayList<>();
        for (Symbol s : symbols) {
            if (s.deleted) continue;
            Integer id = tokenToId.get(s.text);
            if (id != null) {
                ids.add((long) id);
            } else if (unkTokenId >= 0) {
                ids.add((long) unkTokenId);
            }
        }
        return ids;
    }

    private static class Symbol {
        String text;
        Symbol prev, next;
        boolean deleted = false;
        Symbol(String text) { this.text = text; }
    }

    private static class Merge implements Comparable<Merge> {
        final Symbol s1, s2;
        final int rank;
        Merge(Symbol s1, Symbol s2, int rank) {
            this.s1 = s1; this.s2 = s2; this.rank = rank;
        }
        @Override public int compareTo(Merge o) { return Integer.compare(this.rank, o.rank); }
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

    private static boolean detectSentencePieceWhitespace(Map<String, Integer> tokenToId, boolean rawUtf8Vocab) {
        if (!rawUtf8Vocab || tokenToId == null || tokenToId.isEmpty()) {
            return false;
        }
        if (tokenToId.containsKey("▁")) {
            return true;
        }
        int sample = 0;
        int sentencePiece = 0;
        for (String token : tokenToId.keySet()) {
            if (token == null || token.isEmpty()) {
                continue;
            }
            sample++;
            if (token.indexOf('\u2581') >= 0) {
                sentencePiece++;
            }
            if (sample >= 2048) {
                break;
            }
        }
        return sentencePiece > 0;
    }

    @Override
    public int vocabSize() { return tokenToId.size(); }

    @Override
    public int bosTokenId() { return bosTokenId; }

    @Override
    public int eosTokenId() { return eosTokenId; }

    @Override
    public int padTokenId() { return padTokenId; }

    @Override
    public int[] allStopTokenIds() {
        return eosTokenId >= 0 ? new int[]{eosTokenId} : new int[0];
    }

    @Override
    public java.util.Map<String, Integer> specialTokens() {
        return java.util.Collections.unmodifiableMap(specialTokens);
    }
}
