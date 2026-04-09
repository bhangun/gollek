package tech.kayys.gollek.tokenizer.impl;

import tech.kayys.gollek.tokenizer.spi.DecodeOptions;
import tech.kayys.gollek.tokenizer.spi.EncodeOptions;
import tech.kayys.gollek.tokenizer.spi.Tokenizer;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

/**
 * WordPiece tokenizer — BERT-style subword tokenization.
 *
 * <p>Based on the WordPiece algorithm used in BERT, DistilBERT, and RoBERTa.
 * Splits unknown words into known subword pieces prefixed with {@code ##}.
 *
 * <h3>Algorithm</h3>
 * <ol>
 *   <li>Lowercase and whitespace-tokenize input</li>
 *   <li>For each word, greedily find the longest matching prefix in vocab</li>
 *   <li>Remaining suffix is prefixed with {@code ##} and matched again</li>
 * </ol>
 */
public final class WordPieceTokenizer implements Tokenizer {

    private final Map<String, Integer> vocab;
    private final Map<Integer, String> reverseVocab;
    private final int padId, bosId, eosId, unkId;
    private final int maxWordLen;

    public WordPieceTokenizer(Map<String, Integer> vocab,
                               int padId, int bosId, int eosId, int unkId) {
        this.vocab = Collections.unmodifiableMap(vocab);
        this.padId = padId;
        this.bosId = bosId;
        this.eosId = eosId;
        this.unkId = unkId;
        this.maxWordLen = 200;
        this.reverseVocab = new HashMap<>(vocab.size());
        vocab.forEach((k, v) -> reverseVocab.put(v, k));
    }

    public static WordPieceTokenizer fromVocabFile(Path vocabFile) throws IOException {
        List<String> lines = Files.readAllLines(vocabFile);
        Map<String, Integer> vocab = new LinkedHashMap<>(lines.size());
        for (int i = 0; i < lines.size(); i++) vocab.put(lines.get(i).strip(), i);

        int padId = vocab.getOrDefault("[PAD]", 0);
        int bosId = vocab.getOrDefault("[CLS]", 101);
        int eosId = vocab.getOrDefault("[SEP]", 102);
        int unkId = vocab.getOrDefault("[UNK]", 100);
        return new WordPieceTokenizer(vocab, padId, bosId, eosId, unkId);
    }

    @Override
    public long[] encode(String text, EncodeOptions options) {
        if (text == null || text.isEmpty()) return new long[0];
        List<Long> ids = new ArrayList<>();
        
        if (options.addBos && bosId >= 0) {
            ids.add((long) bosId);
        } else {
            // Include CLS token by default as in traditional WordPiece if not overridden
            ids.add((long) bosId);
        }

        for (String word : text.toLowerCase().split("\\s+")) {
            if (word.isEmpty()) continue;
            for (long id : wordPiece(word)) {
                ids.add(id);
            }
        }

        if (options.addEos && eosId >= 0) {
            ids.add((long) eosId);
        } else {
            // Include SEP token by default
            ids.add((long) eosId);
        }

        return ids.stream().mapToLong(Long::longValue).toArray();
    }

    @Override
    public String decode(long[] tokens, DecodeOptions options) {
        StringBuilder sb = new StringBuilder();
        for (long token : tokens) {
            int id = (int) token;
            String tok = reverseVocab.getOrDefault(id, "[UNK]");
            
            if (options.skipSpecialTokens) {
                if (id == bosId || id == eosId || id == padId) continue;
            } else {
                if (tok.equals("[CLS]") || tok.equals("[SEP]") || tok.equals("[PAD]")) continue;
            }

            if (tok.startsWith("##")) {
                sb.append(tok.substring(2));
            } else {
                if (sb.length() > 0) sb.append(' ');
                sb.append(tok);
            }
        }
        return sb.toString().trim();
    }

    @Override
    public int vocabSize() { return vocab.size(); }

    @Override
    public int bosTokenId() { return bosId; }

    @Override
    public int eosTokenId() { return eosId; }

    @Override
    public int padTokenId() { return padId; }

    public int unkTokenId() { return unkId; }

    private List<Long> wordPiece(String word) {
        if (word.length() > maxWordLen) return List.of((long) unkId);
        List<Long> tokens = new ArrayList<>();
        int start = 0;
        while (start < word.length()) {
            int end = word.length();
            String cur = null;
            while (start < end) {
                String substr = (start == 0 ? "" : "##") + word.substring(start, end);
                if (vocab.containsKey(substr)) { cur = substr; break; }
                end--;
            }
            if (cur == null) return List.of((long) unkId); // word not decomposable
            tokens.add((long) vocab.get(cur));
            start = end;
        }
        return tokens;
    }
}
