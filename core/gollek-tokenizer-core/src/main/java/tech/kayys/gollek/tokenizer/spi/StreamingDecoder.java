package tech.kayys.gollek.tokenizer.spi;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;
import java.util.HashSet;
import java.util.Set;

/**
 * Stateful helper for decoding generated tokens into text pieces.
 *
 * <p>Handles the complexity of multi-token BPE merges, UTF-8 byte boundaries,
 * and tokenizer-specific whitespace symbols (e.g., GPT-2's 'Ġ' or SPM's ' ').
 */
public class StreamingDecoder {

    private final Tokenizer tokenizer;
    private final DecodeOptions options;
    private final List<Long> tokens = new ArrayList<>();
    private String lastFullText = "";
    
    private static final Pattern SCRIPT_DETECTOR = Pattern.compile(
        "(?:" +
        "\\p{L}\\p{Lo}|" +  // Kana/CJK
        "\\p{L}\\p{IsGreek}|" +  // Latin + Greek
        "\\p{L}\\p{IsThai}|" +  // Latin + Thai
        "\\p{L}\\p{IsArabic}|" +  // Latin + Arabic
        "\\p{L}\\p{IsDevanagari}" +  // Latin + Devanagari
        ")"
    );
    
    private static final Set<Character> SUSPICIOUS_SEQUENCES = new HashSet<>();
    static {
        // Characters that shouldn't appear together in natural text
        SUSPICIOUS_SEQUENCES.add('\uFFFD');  // Replacement character
    }

    public StreamingDecoder(Tokenizer tokenizer, DecodeOptions options) {
        this.tokenizer = tokenizer;
        this.options = options;
    }

    /**
     * Append a new token and return ONLY the newly decoded text delta.
     *
     * @param tokenId the next token from the model
     * @return the decoded string fragment corresponding to this token
     */
    public String decodeNext(long tokenId) {
        tokens.add(tokenId);

        // We re-decode the full sequence to handle BPE merges and multi-byte UTF-8
        // correctly across token boundaries.
        long[] currentIds = tokens.stream().mapToLong(l -> l).toArray();
        String currentFullText = tokenizer.decode(currentIds, options);
        
        // Early corruption detection
        if (isCorrupted(currentFullText)) {
            System.err.println("StreamingDecoder WARNING: Detected corrupted text at token " + tokenId);
            System.err.println("  Current text: " + debugString(currentFullText));
            System.err.println("  Token ID: " + tokenId);
            System.err.println("  Tokens so far: " + tokens);
            // Return empty to avoid propagating corruption
            return "";
        }

        if (currentFullText.length() < lastFullText.length()) {
            // This can happen with certain special tokens or normalization rules.
            // Reset and return the full text for this step.
            lastFullText = currentFullText;
            return currentFullText;
        }

        String delta = currentFullText.substring(lastFullText.length());
        lastFullText = currentFullText;
        return delta;
    }
    
    private boolean isCorrupted(String text) {
        if (text == null || text.isEmpty()) return false;
        
        int latinCount = 0, nonLatinCount = 0;
        int scriptChanges = 0;
        char prevScript = '\0';
        
        for (char c : text.toCharArray()) {
            if (c >= 0 && c <= 127 && Character.isLetter(c)) {
                latinCount++;
                if (prevScript != 'L') scriptChanges++;
                prevScript = 'L';
            } else if (Character.isLetter(c)) {
                nonLatinCount++;
                if (prevScript != 'N') scriptChanges++;
                prevScript = 'N';
            } else if (c == '\uFFFD') {
                return true;  // Replacement char = corruption
            }
        }
        
        // More than 3 script changes in first 20 chars is suspicious
        if (text.length() <= 20 && scriptChanges > 3) {
            return true;
        }
        
        // Unusual pattern: high script change ratio
        if (scriptChanges > text.length() / 4) {
            return true;
        }
        
        return false;
    }
    
    private String debugString(String text) {
        if (text == null) return "null";
        StringBuilder sb = new StringBuilder();
        for (char c : text.toCharArray()) {
            if (c < 32 || c == 127) {
                sb.append(String.format("\\u%04x", (int)c));
            } else {
                sb.append(c);
            }
        }
        return sb.toString();
    }

    /**
     * Clear the internal state for a new generation session.
     */
    public void reset() {
        tokens.clear();
        lastFullText = "";
    }

    public String currentText() {
        return lastFullText;
    }
}